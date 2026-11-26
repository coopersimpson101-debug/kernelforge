"""Loop-nest passes, run after a schedule template has built the nest.

  lower_reads     tensor Reads/TStores -> flat buffer Loads/Stores; Q8_0/Q4_0/f16 dequantisation is
                  expanded into byte/half loads here; a bounds check (clamped index + select) is
                  inserted only where interval analysis cannot prove the access in range
  unroll          For(kind=unroll) -> straight-line copies (locals renamed per copy)
  vectorize       For(kind=vectorize) -> vec2/vec4 loads, stores and dot-product accumulation; any
                  loop that does not fit the pattern falls back to unrolling
  combine_bytes   groups byte loads at base+c, base+c+1.. into 32-bit (possibly unaligned) word loads
  expand_words    (WGSL only, no 8-bit storage type) words/bytes/halves -> u32 array loads + shifts
  cse             hoists repeated pure sub-expressions (loads of read-only buffers, index maths)
"""
from __future__ import annotations

import math

from . import quant
from .analysis import affine, from_affine, simplify_expr, simplify_kernel, thread_env
from .ir import (Assign, Barrier, Bin, Broadcast, Buffer, ByteLoad, Call, Cast, Comment, Const, Decl,
                 Expr, For, HalfLoad, If, Kernel, Lane, Load, Read, Seq, Select, Stmt, Store, TensorRef,
                 TStore, Var, VecLoad, WordLoad, binop, cast, children, const, elem, flatten_seq, fresh,
                 free_vars, land, lanes, map_stmt, rebuild, select, stmt_exprs, subst, transform, vec,
                 walk)


# ----------------------------------------------------------------------------- env-aware walk

def map_env(s: Stmt, env: dict, fe) -> Stmt:
    """Like map_stmt but ``fe(expr, env)`` sees the ranges of enclosing loops and immutable lets."""
    from .analysis import rng
    if isinstance(s, Seq):
        env = dict(env)
        out = []
        for c in s.body:
            r = map_env(c, env, fe)
            if isinstance(r, Decl) and not r.mutable:
                rr = rng(r.init, env)
                if rr is not None:
                    env[r.var] = rr
            out.append(r)
        return Seq(out)
    if isinstance(s, For):
        e2 = dict(env)
        e2[s.var] = (0, s.extent - 1)
        return For(s.var, s.extent, map_env(s.body, e2, fe), s.kind)
    if isinstance(s, If):
        from .analysis import narrow
        return If(fe(s.cond, env), map_env(s.then, narrow(env, s.cond), fe),
                  map_env(s.els, env, fe) if s.els is not None else None)
    return map_stmt(s, fe=lambda e: fe(e, env))


# ----------------------------------------------------------------------------- lowering of reads

def storage_dtype(t: TensorRef) -> str:
    if t.scope != "global":
        return "u32" if t.dtype == "u32" else "f32"
    return {"f32": "f32", "u32": "u32", "i32": "i32"}.get(t.dtype, "bytes")


def storage_size(t: TensorRef) -> int:
    if t.dtype in quant.BLOCK_BYTES:
        return math.prod(t.shape[:-1]) * (t.shape[-1] // quant.QK) * quant.BLOCK_BYTES[t.dtype]
    if t.dtype == "f16":
        return 2 * t.size
    return t.size


def _flat(idx, shape) -> Expr:
    f = const(0)
    for i, n in zip(idx, shape):
        f = f * const(n) + i
    return f


def lower_reads(k: Kernel) -> Kernel:
    bufs: dict = {}
    written = set(k.outputs) | set(k.alias.values())

    def buf(t: TensorRef) -> Buffer:
        name = k.alias.get(t.name, t.name) if t.scope == "global" else t.name
        if name not in bufs:
            if t.scope == "global":
                bufs[name] = Buffer(name, storage_dtype(t), storage_size(t), "global", name not in written)
            else:
                bufs[name] = Buffer(name, storage_dtype(t), t.size, t.scope, False)
        return bufs[name]

    for p in k.params:
        buf(p)

    def lower(e: Expr, env: dict) -> Expr:
        """Rewrite Reads; the true branch of a Select is lowered under its condition, so guards
        like in_cells(c) ? cell[c] : SOLID prove the inner access in range."""
        from .analysis import narrow
        memo: dict = {}

        def go(x, env):
            key = (x, id(env))
            if key in memo:
                return memo[key]
            if isinstance(x, Select):
                c = go(x.cond, env)
                y = select(c, go(x.t, narrow(env, c)), go(x.f, env))
            else:
                ch = children(x)
                y = rebuild(x, tuple(go(cc, env) for cc in ch)) if ch else x
                r = fn(y, env)
                y = y if r is None else r
            memo[key] = y
            return y

        return go(e, env)

    def fn(x, env):
            if not isinstance(x, Read):
                return None
            t = x.tensor
            b = buf(t)
            if t.scope != "global":
                return Load(b, simplify_expr(_flat(x.idx, t.shape), env), x.dtype)
            shape = list(t.shape)
            if x.part == "d":
                shape[-1] = shape[-1] // quant.QK
            conds, idx = [], []
            for i, n in zip(x.idx, shape):
                c = simplify_expr(land(i >= const(0), i < const(n)), env)
                if not (isinstance(c, Const) and c.value):
                    conds.append(c)
                    idx.append(select(c, i, const(0)))  # clamp only the unproven dimension
                else:
                    idx.append(i)
            inb = land(*conds) if conds else const(True)
            idx = tuple(idx)
            val = _element(b, t, idx, x.part)
            if conds:
                val = select(inb, val, const(0, val.dtype))
            return simplify_expr(val, env)

    def fs_env(s: Stmt, env: dict) -> Stmt:
        if isinstance(s, TStore):
            b = buf(s.tensor)
            return Store(b, simplify_expr(_flat(s.idx, s.tensor.shape), env), lower(s.value, env))
        return map_stmt(s, fe=lambda e: lower(e, env))

    def go(s: Stmt, env: dict) -> Stmt:
        from .analysis import rng
        if isinstance(s, Seq):
            env = dict(env)
            out = []
            for c in s.body:
                r = go(c, env)
                if isinstance(r, Decl) and not r.mutable:
                    rr = rng(r.init, env)
                    if rr is not None:
                        env[r.var] = rr
                out.append(r)
            return Seq(out)
        if isinstance(s, For):
            e2 = dict(env)
            e2[s.var] = (0, s.extent - 1)
            return For(s.var, s.extent, go(s.body, e2), s.kind)
        if isinstance(s, If):
            from .analysis import narrow
            return If(lower(s.cond, env), go(s.then, narrow(env, s.cond)), go(s.els, env) if s.els is not None else None)
        return fs_env(s, env)

    k.body = go(k.body, thread_env(k))
    k.buffers = bufs
    k.allocs = [b for b in bufs.values() if b.scope != "global"]
    return k


def _element(b: Buffer, t: TensorRef, idx: tuple, part: str) -> Expr:
    if t.dtype in ("f32", "u32", "i32"):
        return Load(b, _flat(idx, t.shape), "f32" if t.dtype == "f32" else t.dtype)
    if t.dtype == "f16":
        return HalfLoad(b, _flat(idx, t.shape) * const(2))
    QK, BB = quant.QK, quant.BLOCK_BYTES[t.dtype]
    nb = t.shape[-1] // QK
    row = _flat(idx[:-1], t.shape[:-1]) if len(idx) > 1 else const(0)
    if part == "d":
        return HalfLoad(b, (row * const(nb) + idx[-1]) * const(BB))
    kk = idx[-1]
    base = (row * const(nb) + kk / const(QK)) * const(BB)
    j = kk % const(QK)
    if t.dtype == "q8_0":
        q = Cast(Call("sext8", (ByteLoad(b, base + const(2) + j),), "i32"), "f32")
    else:
        byte = ByteLoad(b, base + const(2) + j % const(16))
        sh = cast((j / const(16)) * const(4), "u32")
        nib = (byte >> sh) & const(15, "u32")
        q = Cast(cast(nib, "i32"), "f32") - const(8.0)
    if part == "q":
        return q
    return HalfLoad(b, base) * q


# ----------------------------------------------------------------------------- unroll

def _rename_decls(s: Stmt) -> Stmt:
    names = {}

    def collect(t):
        if isinstance(t, Seq):
            for c in t.body:
                collect(c)
        elif isinstance(t, For):
            collect(t.body)
        elif isinstance(t, If):
            collect(t.then)
            if t.els is not None:
                collect(t.els)
        elif isinstance(t, Decl):
            names[t.var] = fresh(t.var.name.rstrip("0123456789") or "v", t.var.dtype)
    collect(s)
    if not names:
        return s

    def fs(t):
        if isinstance(t, Decl):
            return Decl(names.get(t.var, t.var), t.init, t.mutable)
        if isinstance(t, Assign):
            return Assign(names.get(t.var, t.var), t.value)
        return None
    return map_stmt(s, fs=fs, fe=lambda e: subst(e, names))


def _unroll_for(s: For) -> Stmt:
    copies = []
    for i in range(s.extent):
        c = map_stmt(s.body, fe=lambda e, i=i: subst(e, {s.var: const(i)}))
        copies.append(_rename_decls(c))
    return Seq(copies)


def unroll(k: Kernel) -> Kernel:
    def fs(t):
        if isinstance(t, For) and t.kind == "unroll":
            return _unroll_for(t)
        return None
    k.body = flatten_seq(map_stmt(k.body, fs=fs))
    return k


# ----------------------------------------------------------------------------- vectorize

class _NoVec(Exception):
    pass


def _vectorize_for(s: For) -> Stmt:
    v, n = s.var, s.extent
    vlets: dict = {}

    def dep(e):
        fv = free_vars(e)
        return v in fv or bool(fv & set(vlets))

    def coef(idx):
        af = affine(idx)
        if af is None:
            raise _NoVec
        terms, _ = af
        for atom, c in terms.items():
            if atom != v and v in free_vars(atom):
                raise _NoVec
        return terms.get(v, 0)

    def vx(e):
        if not dep(e):
            return e
        if isinstance(e, Var):
            if e in vlets:
                return vlets[e]
            raise _NoVec
        if isinstance(e, Load):
            if coef(e.idx) != 1 or e.dtype != "f32":
                raise _NoVec
            return VecLoad(e.buf, subst(e.idx, {v: const(0)}), n, vec("f32", n))
        if isinstance(e, Bin) and e.op in ("+", "-", "*", "/", "min", "max") and elem(e.dtype) == "f32":
            a, b = vx(e.a), vx(e.b)
            if lanes(a.dtype) == 1:
                a = Broadcast(a, n, vec("f32", n))
            if lanes(b.dtype) == 1:
                b = Broadcast(b, n, vec("f32", n))
            return Bin(e.op, a, b, vec("f32", n))
        if isinstance(e, Call) and e.fn in ("exp", "sqrt", "rsqrt", "tanh", "abs") and e.dtype == "f32":
            return Call(e.fn, tuple(vx(a) for a in e.args), vec("f32", n))
        raise _NoVec

    def vs(t: Stmt) -> Stmt:
        if isinstance(t, Seq):
            return Seq([vs(c) for c in t.body])
        if isinstance(t, Decl):
            if not dep(t.init):
                return t
            if t.mutable:
                raise _NoVec
            nv = fresh(t.var.name.rstrip("0123456789"), vec("f32", n))
            val = vx(t.init)
            vlets[t.var] = nv
            return Decl(nv, val, False)
        if isinstance(t, Assign):
            if not dep(t.value):
                raise _NoVec  # same scalar assigned n times: leave to unroll
            x = t.value
            if isinstance(x, Bin) and x.op == "+" and x.a == t.var and not dep(x.a):
                f = x.b
                if isinstance(f, Bin) and f.op == "*":
                    a, b = vx(f.a), vx(f.b)
                    if lanes(a.dtype) == n and lanes(b.dtype) == n:
                        return Assign(t.var, t.var + Call("dot", (a, b), "f32"))
                fv = vx(f)
                if lanes(fv.dtype) == 1:
                    raise _NoVec
                return Assign(t.var, t.var + Call("hsum", (fv,), "f32"))
            raise _NoVec
        if isinstance(t, Store):
            if coef(t.idx) != 1:
                raise _NoVec
            val = vx(t.value)
            if lanes(val.dtype) == 1:
                val = Broadcast(val, n, vec(val.dtype, n))
            return Store(t.buf, subst(t.idx, {v: const(0)}), val)
        if isinstance(t, If):
            if dep(t.cond):
                raise _NoVec
            return If(t.cond, vs(t.then), vs(t.els) if t.els is not None else None)
        if isinstance(t, Comment):
            return t
        raise _NoVec

    try:
        return vs(s.body)
    except _NoVec:
        return _unroll_for(For(s.var, s.extent, s.body, "unroll"))


def vectorize(k: Kernel) -> Kernel:
    def fs(t):
        if isinstance(t, For) and t.kind == "vectorize":
            return _vectorize_for(t)
        return None
    k.body = flatten_seq(map_stmt(k.body, fs=fs))
    return k


# ----------------------------------------------------------------------------- bytes -> words

def combine_bytes(k: Kernel) -> Kernel:
    loads = {x for e in stmt_exprs(k.body) for x in walk(e) if isinstance(x, ByteLoad)}
    groups: dict = {}
    for x in loads:
        af = affine(x.off)
        if af is None:
            continue
        terms, c = af
        key = (x.buf, frozenset(terms.items()))
        groups.setdefault(key, []).append((c, x, terms))
    m = {}
    for (b, _), items in groups.items():
        cs = sorted({c for c, _, _ in items})
        if len(cs) < 2:
            continue
        c0 = cs[0]
        for c, x, terms in items:
            chunk = (c - c0) // 4
            w = WordLoad(b, from_affine(terms, c0 + 4 * chunk))
            sh = const(8 * ((c - c0) % 4), "u32")
            m[x] = (w >> sh) & const(255, "u32") if sh.value else w & const(255, "u32")
    if m:
        k.body = map_stmt(k.body, fe=lambda e: transform(e, lambda x: m.get(x) if isinstance(x, ByteLoad) else None))
    return k


def slp_loads(k: Kernel) -> Kernel:
    """Superword-level vectorisation of loads: scalar f32 loads at base+c (base a multiple of 4)
    that share an aligned 4-element group become lanes of one 128-bit vector load."""
    from .analysis import let_defs, multiple_of
    defs = let_defs(k.body)
    loads = {x for e in stmt_exprs(k.body) for x in walk(e)
             if isinstance(x, Load) and x.buf.scope == "global" and x.dtype == "f32" and x.buf.readonly
             and x.buf.size % 4 == 0}
    groups: dict = {}
    for x in loads:
        af = affine(x.idx)
        if af is None:
            continue
        terms, c = af
        if terms and multiple_of(from_affine(terms, 0), defs) % 4:
            continue
        groups.setdefault((x.buf, frozenset(terms.items()), c // 4), []).append((c, x, terms))
    m = {}
    for (b, _, q), items in groups.items():
        if len({c for c, _, _ in items}) < 2:
            continue
        for c, x, terms in items:
            m[x] = Lane(VecLoad(b, from_affine(terms, 4 * q), 4, "f32x4"), c - 4 * q, "f32")
    if m:
        k.body = map_stmt(k.body, fe=lambda e: transform(e, lambda x: m.get(x) if isinstance(x, Load) else None))
    return k


def expand_words(k: Kernel) -> Kernel:
    """WGSL has no 8/16-bit storage: byte buffers are array<u32>; rebuild every narrow load."""
    def word(b, o):
        return Load(b, o / const(4), "u32")

    def shift(o):
        return cast((o % const(4)) * const(8), "u32")

    def fn(x):
        if isinstance(x, WordLoad):
            w0 = word(x.buf, x.off)
            w1 = word(x.buf, x.off + const(4))
            s = shift(x.off)
            if isinstance(s, Const):
                if s.value == 0:
                    return w0
                return (w0 >> s) | (w1 << const(32 - s.value, "u32"))
            return select(s.eq(const(0, "u32")), w0, (w0 >> s) | (w1 << (const(32, "u32") - s)))
        if isinstance(x, ByteLoad):
            return (word(x.buf, x.off) >> shift(x.off)) & const(255, "u32")
        if isinstance(x, HalfLoad):
            return Call("unpack_lo_half", (word(x.buf, x.off) >> shift(x.off),), "f32")
        return None
    k.body = map_stmt(k.body, fe=lambda e: transform(e, fn))
    return k


# ----------------------------------------------------------------------------- CSE

def _mutables(s: Stmt) -> set:
    out = set()

    def go(t):
        if isinstance(t, Seq):
            for c in t.body:
                go(c)
        elif isinstance(t, For):
            go(t.body)
        elif isinstance(t, If):
            go(t.then)
            if t.els is not None:
                go(t.els)
        elif isinstance(t, Decl) and t.mutable:
            out.add(t.var)
    go(s)
    return out


def _size(e):
    return sum(1 for _ in walk(e))


def cse(k: Kernel) -> Kernel:
    mut = _mutables(k.body)
    writable = {b for b in k.buffers.values() if not b.readonly}

    def pure(e):
        for x in walk(e):
            if isinstance(x, Var) and x in mut:
                return False
            if isinstance(x, (Load, VecLoad, ByteLoad, WordLoad, HalfLoad)) and x.buf in writable:
                return False
            if isinstance(x, Call) and x.fn.startswith("subgroup"):
                return False
        return True

    def top_exprs(t):
        if isinstance(t, (Decl,)): return [t.init]
        if isinstance(t, Assign): return [t.value]
        if isinstance(t, Store): return [t.idx, t.value]
        if isinstance(t, If): return [t.cond]
        return []

    def block(seq: Seq) -> Seq:
        body = [rec(c) for c in seq.body]
        counts: dict = {}
        first: dict = {}
        uncond: set = set()  # evaluated unconditionally somewhere (not only inside a Select arm)

        def count(e, pos, cond):
            if isinstance(e, (Const, Var)):
                return
            counts[e] = counts.get(e, 0) + 1
            first.setdefault(e, pos)
            seen_uncond = e in uncond
            if not cond:
                uncond.add(e)
            if counts[e] == 1 or (not cond and not seen_uncond):
                if isinstance(e, Select):
                    count(e.cond, pos, cond)
                    count(e.t, pos, True)
                    count(e.f, pos, True)
                else:
                    for c in children(e):
                        count(c, pos, cond)

        for pos, t in enumerate(body):
            for e in top_exprs(t):
                count(e, pos, False)

        def has_load(e):
            return any(isinstance(x, (Load, VecLoad, WordLoad, HalfLoad, ByteLoad)) for x in walk(e))

        # hoisting moves evaluation out of Select arms: only allowed for loads that are evaluated
        # unconditionally anyway (C/MSL ternaries are lazy and may be guarding an access)
        cands = [e for e, c in counts.items() if c >= 2 and pure(e)
                 and (_size(e) >= 3 or isinstance(e, (Load, VecLoad, WordLoad, HalfLoad, ByteLoad)))
                 and (e in uncond or not has_load(e))]
        if not cands:
            return Seq(body)
        cands = list(dict.fromkeys(cands))
        cands.sort(key=_size)
        names = {e: fresh("c", e.dtype) for e in cands}

        def rep(e, skip=None):
            if e in names and e is not skip and e != skip:
                return names[e]
            ch = children(e)
            return rebuild(e, tuple(rep(c) for c in ch)) if ch else e

        inserts: dict = {}
        for e in cands:
            inserts.setdefault(first[e], []).append(Decl(names[e], rep(e, skip=e), False))
        out = []
        for pos, t in enumerate(body):
            out.extend(inserts.get(pos, []))
            out.append(map_stmt_shallow(t, rep))
        return Seq(out)

    def map_stmt_shallow(t, rep):
        if isinstance(t, Decl): return Decl(t.var, rep(t.init), t.mutable)
        if isinstance(t, Assign): return Assign(t.var, rep(t.value))
        if isinstance(t, Store): return Store(t.buf, rep(t.idx), rep(t.value))
        if isinstance(t, If): return If(rep(t.cond), t.then, t.els)
        return t

    def rec(t):
        if isinstance(t, Seq): return block(t)
        if isinstance(t, For): return For(t.var, t.extent, rec(t.body) if isinstance(t.body, Seq) else block(Seq([t.body])), t.kind)
        if isinstance(t, If):
            th = rec(t.then) if isinstance(t.then, Seq) else block(Seq([t.then]))
            el = None if t.els is None else (rec(t.els) if isinstance(t.els, Seq) else block(Seq([t.els])))
            return If(t.cond, th, el)
        return t

    k.body = rec(k.body if isinstance(k.body, Seq) else Seq([k.body]))
    return k


# ----------------------------------------------------------------------------- pipeline

def finalize(k: Kernel, target_name: str) -> Kernel:
    if any(isinstance(x, Call) and x.fn.startswith("subgroup") for e in stmt_exprs(k.body) for x in walk(e)):
        k.features.add("subgroups")
    k = lower_reads(k)
    k = simplify_kernel(k)
    k = unroll(k)
    k = simplify_kernel(k)
    k = vectorize(k)
    k = simplify_kernel(k)
    if target_name == "c":
        k = cse(k)
        return k
    k = combine_bytes(k)
    k = slp_loads(k)
    if target_name == "wgsl":
        k = expand_words(k)
    k = simplify_kernel(k)
    k = cse(k)
    k.body = flatten_seq(k.body)
    return k
