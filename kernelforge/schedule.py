"""Schedules (the "how"): turn a ComputeDef into a loop nest, given knob values.

This is the TVM split between *compute* and *schedule*. Each pattern kind has a parameterised
schedule template (as in TVM's TOPI/AutoTVM templates): the template decides the thread/block
mapping and tiling and then applies generic primitives:

* ``lower_reduces``  - materialise ``Reduce`` nodes as accumulator loops (optionally unrolled);
* ``stage``          - cache_read: stage every read of a tensor inside a region into a shared
                       (cooperative) or register tile. The tile footprint comes from **bound
                       inference** on the affine index expressions: loop variables inside the region
                       (and, for shared tiles, the thread indices) are relaxed over their ranges,
                       everything else stays symbolic as the tile origin;
* loop kinds ``unroll`` / ``vectorize`` that the passes in ``passes.py`` act on.

Knob spaces for the autotuner are in ``space()``; ``default_knobs()`` is the untuned baseline.
"""
from __future__ import annotations

import itertools
import math

from .analysis import affine, from_affine
from .compute import ACC, ComputeDef
from .ir import (BX, BY, BZ, TX, TY, TZ, Barrier, Bin, Builder, Select, Call, Const, Expr, For, If, Kernel, Read,
                 Reduce, Seq, Stmt, TensorRef, TStore, Var, binop, children, const, emax, free_vars,
                 fresh, land, map_stmt, stmt_exprs, subst, transform, walk)
from .target import Target

NEG_BIG = -3.4e38


class ScheduleError(Exception):
    pass


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ----------------------------------------------------------------------------- primitives

def loop_on(b: Builder, v: Var, extent: int, kind: str = "serial"):
    class _L:
        def __enter__(self_):
            b.push()
            return v

        def __exit__(self_, *a):
            body = b.pop()
            if extent == 1:
                b.emit(map_stmt(body, fe=lambda e: subst(e, {v: const(0)})))
            elif extent > 1:
                b.emit(For(v, extent, body, kind))
    return _L()


def combine(op: str, a: Expr, b: Expr) -> Expr:
    return a + b if op == "sum" else emax(a, b)


def lower_reduces(b: Builder, e: Expr, kind: str = "serial") -> Expr:
    """Emit accumulator loops for every Reduce in ``e`` (outermost first, nesting inner ones)."""
    tops: list = []

    def find(x):
        if isinstance(x, Reduce):
            if x not in tops:
                tops.append(x)
            return
        for c in children(x):
            find(c)

    find(e)
    m = {}
    for r in tops:
        acc = b.var("acc", const(0.0) if r.op == "sum" else const(NEG_BIG))

        def nest(axes):
            if not axes:
                val = lower_reduces(b, r.body, kind)
                b.assign(acc, combine(r.op, acc, val))
                return
            v, n = axes[0]
            with loop_on(b, v, n, kind):
                nest(axes[1:])

        nest(list(r.axes))
        m[r] = acc
    return transform(e, lambda x: m.get(x) if isinstance(x, Reduce) else None)


def loops_in(s: Stmt) -> dict:
    out = {}
    if isinstance(s, Seq):
        for c in s.body:
            out.update(loops_in(c))
    elif isinstance(s, For):
        out[s.var] = s.extent
        out.update(loops_in(s.body))
    elif isinstance(s, If):
        out.update(loops_in(s.then))
        if s.els is not None:
            out.update(loops_in(s.els))
    return out


def _footprint(e: Expr, relaxed: dict):
    af = affine(e)
    if af is None:
        raise ScheduleError(f"non-affine index {e}")
    terms, c = af
    fixed, lo, span = {}, c, 0
    for atom, coef in terms.items():
        if isinstance(atom, Var) and atom in relaxed:
            ext = relaxed[atom]
            if coef < 0:
                lo += coef * (ext - 1)
            span += abs(coef) * (ext - 1)
        else:
            if free_vars(atom) & set(relaxed):
                raise ScheduleError("relaxed var inside non-affine index term")
            fixed[atom] = coef
    return fixed, lo, lo + span


def stage(region: Stmt, tensor: TensorRef, scope: str, name: str, threads: dict | None = None,
          part: str = "", vec: int = 1, pad: int = 0, in_loop: bool = True):
    """cache_read: stage all reads of ``tensor`` in ``region`` into a tile. Returns the new region.

    ``threads`` = {Var: extent} of the thread indices that cooperate (shared scope); they are relaxed
    together with every loop inside the region. Raises ScheduleError if the footprint is not a box
    with a common symbolic origin (then the template simply does not stage this tensor).
    """
    relaxed = dict(loops_in(region))
    if scope == "shared":
        relaxed.update(threads or {})
    sites = [x for e in stmt_exprs(region) for x in walk(e)
             if isinstance(x, Read) and x.tensor == tensor and x.part == part]
    if not sites:
        raise ScheduleError(f"no reads of {tensor.name}")
    nd = len(tensor.shape)
    fixed = [None] * nd
    lo = [None] * nd
    hi = [None] * nd
    for r in sites:
        for d in range(nd):
            f, l, h = _footprint(r.idx[d], relaxed)
            if fixed[d] is None:
                fixed[d], lo[d], hi[d] = f, l, h
            elif fixed[d] != f:
                raise ScheduleError("reads with different symbolic origins")
            else:
                lo[d], hi[d] = min(lo[d], l), max(hi[d], h)
    for d in range(nd):
        if not fixed[d] and tensor.scope == "global":  # constant origin: clip to the tensor
            lo[d], hi[d] = max(lo[d], 0), min(hi[d], tensor.shape[d] - 1)
    ext = [hi[d] - lo[d] + 1 for d in range(nd)]
    base = [from_affine(fixed[d], lo[d]) for d in range(nd)]
    tshape = tuple(ext[:-1]) + (ext[-1] + pad,)
    tile = TensorRef(name, tshape, "u32" if sites[0].dtype == "u32" else "f32", scope)
    dt = sites[0].dtype

    def rw(x):
        if isinstance(x, Read) and x.tensor == tensor and x.part == part:
            return Read(tile, tuple(binop("-", i, bs) for i, bs in zip(x.idx, base)), "", dt)
        return None

    new_region = map_stmt(region, fe=lambda e: transform(e, rw))
    b = Builder()
    total = math.prod(ext)
    if scope == "shared":
        tid, nthreads = _flat_tid(threads)
        if ext[-1] % vec:
            vec = 1
        units = total // vec
        steps = cdiv(units, nthreads)
        with b.loop("s", steps, "unroll" if steps <= 16 else "serial") as s:
            u = s * const(nthreads) + tid
            with b.if_(u < const(units)):
                with b.loop("v", vec, "vectorize") as v:
                    f = u * const(vec) + v
                    d, rem = [], f
                    for k in reversed(range(nd)):
                        d.append(rem % const(ext[k]) if k else rem)
                        rem = rem / const(ext[k])
                    d = tuple(reversed(d))
                    src = Read(tensor, tuple(bs + dd for bs, dd in zip(base, d)), part, dt)
                    b.emit(TStore(tile, d, src))
        load = b.build()
        if not in_loop:
            return Seq([load, Barrier(), new_region]), tile
        return Seq([load, Barrier(), new_region, Barrier()]), tile
    # register / local tile: every thread loads its own footprint
    idx = []

    def nest(k):
        if k == nd:
            src = Read(tensor, tuple(bs + dd for bs, dd in zip(base, idx)), part, dt)
            b.emit(TStore(tile, tuple(idx), src))
            return
        with b.loop("r", ext[k], "unroll") as v:
            idx.append(v)
            nest(k + 1)
            idx.pop()

    nest(0)
    return Seq([b.build(), new_region]), tile


def _flat_tid(threads: dict):
    order = [v for v in (TX, TY, TZ) if v in threads]
    tid, mul = const(0), 1
    for v in order:
        tid = tid + v * const(mul)
        mul *= threads[v]
    return tid, mul


def _decompose(f: Expr, shape: tuple) -> list:
    out, rem = [], f
    for k in reversed(range(len(shape))):
        out.append(rem % const(shape[k]) if k else rem)
        rem = rem / const(shape[k])
    return list(reversed(out))


def _conjuncts(c: Expr) -> list:
    if isinstance(c, Bin) and c.op == "&&":
        return _conjuncts(c.a) + _conjuncts(c.b)
    return [c]


def _store(b: Builder, cd: ComputeDef, idx: list, value: Expr):
    """Store with control flow instead of data flow where it saves work: a store predicate
    a && b && c becomes nested ifs (cheap tests first, as written), and a top-level
    select(c, t, f) value becomes if/else, so the expensive arm is only computed when taken."""
    m = dict(zip(cd.axes, idx))
    v = subst(value, m)
    conds = _conjuncts(subst(cd.predicate, m)) if cd.predicate is not None else []

    def emit_value(val):
        if isinstance(val, Select) and not isinstance(val.cond, Const):
            b.push()
            b.emit(TStore(cd.out, tuple(idx), val.t))
            th = b.pop()
            b.push()
            b.emit(TStore(cd.out, tuple(idx), val.f))
            b.emit(If(val.cond, th, b.pop()))
        else:
            b.emit(TStore(cd.out, tuple(idx), val))

    def nest(cs):
        if not cs:
            emit_value(v)
            return
        with b.if_(cs[0]):
            nest(cs[1:])
    nest(conds)


def _grid1d(n: int, max_groups: int):
    if n <= max_groups:
        return (n, 1, 1), BX
    gx = max_groups
    gy = cdiv(n, gx)
    return (gx, gy, 1), BY * const(gx) + BX


def _kernel(cd: ComputeDef, body: Stmt, wg, grid, knobs, features=()):
    params = list(cd.inputs) + ([cd.out] if cd.out.name not in cd.alias else [])
    k = Kernel(cd.name, params, [cd.out.name], body, tuple(wg), tuple(grid), features=set(features),
               meta={"knobs": dict(knobs), "kind": cd.kind}, alias=dict(cd.alias))
    return k


# ----------------------------------------------------------------------------- templates

def cpu_schedule(cd: ComputeDef) -> Kernel:
    """Plain serial loop nest (C oracle): no threads, no tiling."""
    b = Builder()
    shape = cd.out.shape
    if cd.kind == "row":
        lead = list(cd.axes[:-1])
        with _nest(b, lead, shape[:-1]):
            env = {}
            for v, red in cd.reductions:
                val = lower_reduces(b, subst(red, env))
                env[v] = b.let("rv", val)
            with loop_on(b, cd.axes[-1], shape[-1]):
                _store(b, cd, list(cd.axes), subst(cd.body, env))
    else:
        with _nest(b, list(cd.axes), shape):
            if cd.reduce is not None:
                acc = lower_reduces(b, cd.reduce)
                val = subst(cd.body, {ACC: acc})
            else:
                val = cd.body
            _store(b, cd, list(cd.axes), val)
    return _kernel(cd, b.build(), (1, 1, 1), (1, 1, 1), {})


class _nest:
    def __init__(self, b, axes, shape):
        self.cms = [loop_on(b, v, n) for v, n in zip(axes, shape)]

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


def injective_schedule(cd: ComputeDef, kn: dict, t: Target) -> Kernel:
    wg, vec, ept = kn["wg"], kn["vec"], kn["ept"]
    shape = cd.out.shape
    total = math.prod(shape)
    if vec > 1 and shape[-1] % vec:
        raise ScheduleError("vector width must divide the last dim")
    groups = cdiv(total, wg * vec * ept)
    grid, bid = _grid1d(groups, t.max_groups)
    b = Builder()
    with b.loop("e", ept, "unroll") as e:
        g = (bid * const(ept) + e) * const(wg) + TX
        with b.if_(g * const(vec) < const(total)):
            with b.loop("v", vec, "vectorize") as v:
                f = g * const(vec) + v
                _store(b, cd, _decompose(f, shape), cd.body)
    return _kernel(cd, b.build(), (wg, 1, 1), grid, kn)


def stencil_schedule(cd: ComputeDef, kn: dict, t: Target) -> Kernel:
    wx, wy, wz, cz = kn["wx"], kn["wy"], kn["wz"], kn["cz"]
    Z, Y, X = cd.out.shape
    grid = (cdiv(X, wx), cdiv(Y, wy), cdiv(Z, wz * cz))
    b = Builder()
    with b.loop("c", cz, "unroll") as c:
        x = BX * const(wx) + TX
        y = BY * const(wy) + TY
        z = BZ * const(wz * cz) + c * const(wz) + TZ
        with b.if_(land(x < const(X), y < const(Y), z < const(Z))):
            _store(b, cd, [z, y, x], cd.body)
    body = b.build()
    if kn.get("stage"):
        threads = {TX: wx, TY: wy, TZ: wz}
        k = 0
        for tns in list(cd.inputs):
            if len(tns.shape) != 3:
                continue
            reads = [r for e in stmt_exprs(body) for r in walk(e) if isinstance(r, Read) and r.tensor == tns]
            if len({r.idx for r in reads}) <= 1:
                continue  # only read at the centre: staging cannot help
            body, _ = stage(body, tns, "shared", f"sh{k}", threads, in_loop=False)
            k += 1
    return _kernel(cd, body, (wx, wy, wz), grid, kn)


def _block_reduce(b: Builder, x: Expr, op: str, tid: Expr, n: int, t: Target, mode: str, name: str) -> Expr:
    """Combine ``x`` across ``n`` threads (power of two); result visible to all threads."""
    sg = t.subgroup_size
    fn = "subgroup_add" if op == "sum" else "subgroup_max"
    if mode == "subgroup" and sg and n % sg == 0:
        r = b.let("sg", Call(fn, (x,), "f32"))
        if n == sg:
            return r
        nw = n // sg
        sm = TensorRef(name, (nw,), "f32", "shared")
        with b.if_((tid % const(sg)).eq(const(0))):
            b.emit(TStore(sm, (tid / const(sg),), r))
        b.barrier()
        acc = b.var("bred", Read(sm, (const(0),)))
        with b.loop("w", nw - 1, "unroll") as w:
            b.assign(acc, combine(op, acc, Read(sm, (w + const(1),))))
        b.barrier()
        return acc
    sm = TensorRef(name, (n,), "f32", "shared")
    b.emit(TStore(sm, (tid,), x))
    b.barrier()
    s = n // 2
    while s >= 1:
        with b.if_(tid < const(s)):
            b.emit(TStore(sm, (tid,), combine(op, Read(sm, (tid,)), Read(sm, (tid + const(s),)))))
        b.barrier()
        s //= 2
    r = b.let("bred", Read(sm, (const(0),)))
    b.barrier()
    return r


def row_schedule(cd: ComputeDef, kn: dict, t: Target) -> Kernel:
    wg, mode = kn["wg"], kn["xlane"]
    shape = cd.out.shape
    rows_shape = shape[:-1]
    nrows = math.prod(rows_shape) if rows_shape else 1
    if nrows > t.max_groups:
        raise ScheduleError("too many rows for a 1-D grid")
    b = Builder()
    lead = _decompose(BX, rows_shape) if rows_shape else []
    env = dict(zip(cd.axes[:-1], lead))
    for ri, (v, red) in enumerate(cd.reductions):
        red = subst(red, env)
        if not isinstance(red, Reduce):
            env[v] = b.let("rv", red)
            continue
        (j, M), = red.axes
        part = b.var("part", const(0.0) if red.op == "sum" else const(NEG_BIG))
        with b.loop("it", cdiv(M, wg)) as it:
            jj = it * const(wg) + TX
            with b.if_(jj < const(M)):
                b.assign(part, combine(red.op, part, subst(red.body, {j: jj})))
        env[v] = _block_reduce(b, part, red.op, TX, wg, t, mode, f"red{ri}")
    N = shape[-1]
    with b.loop("it", cdiv(N, wg)) as it:
        jj = it * const(wg) + TX
        with b.if_(jj < const(N)):
            _store(b, cd, lead + [jj], subst(cd.body, env))
    body = b.build()
    if kn.get("stage"):
        reds = [r for _, r in cd.reductions if isinstance(r, Reduce)]
        M0 = reds[0].axes[0][1] if reds else -1
        for k, tns in enumerate(cd.inputs):
            reads = [r for e in stmt_exprs(body) for r in walk(e) if isinstance(r, Read) and r.tensor == tns]
            if len(reads) >= 2 and tns.shape[-1] == M0 and 4 * M0 * (k + 1) <= t.max_shared // 2:
                try:
                    body, _ = stage(body, tns, "shared", f"row{k}", {TX: wg}, in_loop=False)
                except ScheduleError:
                    pass
    return _kernel(cd, body, (wg, 1, 1), (nrows, 1, 1), kn)


def matvec_schedule(cd: ComputeDef, kn: dict, t: Target) -> Kernel:
    R, T, U, V, mode = kn["R"], kn["T"], kn["U"], kn["V"], kn["xlane"]
    N = cd.meta["N"]
    red = cd.reduce
    (k0, KE), = red.axes
    if V > 1 and (cd.meta["form"] == "blocked" or KE % V):
        raise ScheduleError("vectorised matvec needs a flat f32/f16 reduction with K % V == 0")
    b = Builder()
    n = BX * const(R) + TY
    jaxis = cd.axes[-1]
    m = {jaxis: n}
    if len(cd.axes) == 2:
        m[cd.axes[0]] = const(0)
    body0 = subst(red.body, m)
    acc = b.var("acc", const(0.0))
    per = T * V * U
    with b.loop("it", cdiv(KE, per)) as it:
        b.push()
        with b.loop("u", U, "unroll") as u:
            kb = ((it * const(U) + u) * const(T) + TX) * const(V)
            with b.if_(kb < const(KE)):
                with b.loop("v", V, "vectorize") as v:
                    val = lower_reduces(b, subst(body0, {k0: kb + v}), "unroll" if kn.get("unroll", 1) else "serial")
                    b.assign(acc, acc + val)
        region = b.pop()
        if kn.get("stage_x"):
            x = next((tn for tn in cd.inputs if len(tn.shape) <= 2 and tn.shape[-1] == cd.meta["K"]
                      and tn.dtype == "f32" and tn.name != cd.out.name and math.prod(tn.shape) == cd.meta["K"]),
                     None)
            if x is None:
                raise ScheduleError("no vector operand to stage")
            region, _ = stage(region, x, "shared", "xs", {TX: T, TY: R}, vec=4 if V == 4 else 1)
        b.emit(region)
    tid = TX
    if T > 1:
        if mode == "subgroup" and t.subgroup_size == T:
            total = b.let("sum", Call("subgroup_add", (acc,), "f32"))
        else:
            sm = TensorRef("red", (R, T), "f32", "shared")
            b.emit(TStore(sm, (TY, tid), acc))
            b.barrier()
            s = T // 2
            while s >= 1:
                with b.if_(tid < const(s)):
                    b.emit(TStore(sm, (TY, tid), Read(sm, (TY, tid)) + Read(sm, (TY, tid + const(s)))))
                b.barrier()
                s //= 2
            total = b.let("sum", Read(sm, (TY, const(0))))
    else:
        total = acc
    with b.if_(land(tid.eq(const(0)), n < const(N))):
        idx = [const(0), n] if len(cd.axes) == 2 else [n]
        _store(b, cd, idx, subst(cd.body, {ACC: total}))
    feats = {"subgroups"} if (T > 1 and mode == "subgroup" and t.subgroup_size == T) else set()
    return _kernel(cd, b.build(), (T, R, 1), (cdiv(N, R), 1, 1), kn, feats)


def matmul_schedule(cd: ComputeDef, kn: dict, t: Target) -> Kernel:
    BM, BN, BK, TM, TN = kn["BM"], kn["BN"], kn["BK"], kn["TM"], kn["TN"]
    M, N, K = cd.meta["M"], cd.meta["N"], cd.meta["K"]
    red = cd.reduce
    (k0, KE), = red.axes
    if BM % TM or BN % TN:
        raise ScheduleError("thread tile must divide block tile")
    wx, wy = BN // TN, BM // TM
    b = Builder()
    accT = TensorRef("acc", (TM, TN), "f32", "local")
    with b.loop("ii", TM, "unroll") as ii:
        with b.loop("jj", TN, "unroll") as jj:
            b.emit(TStore(accT, (ii, jj), const(0.0)))

    def I(ii): return BY * const(BM) + ii * const(wy) + TY
    def J(jj): return BX * const(BN) + jj * const(wx) + TX

    with b.loop("ko", cdiv(K, BK)) as ko:
        b.push()
        with b.loop("ki", BK, "unroll" if kn.get("unroll", 1) else "serial") as ki:
            b.push()
            kk = ko * const(BK) + ki
            with b.if_(kk < const(K)):
                with b.loop("ii", TM, "unroll") as ii:
                    with b.loop("jj", TN, "unroll") as jj:
                        m = {cd.axes[0]: I(ii), cd.axes[1]: J(jj), k0: kk}
                        val = lower_reduces(b, subst(red.body, m), "unroll")
                        b.emit(TStore(accT, (ii, jj), Read(accT, (ii, jj)) + val))
            kbody = b.pop()
            b.emit(kbody)
        region = b.pop()
        tiles = {}
        if kn.get("stage", 1):
            for q, tns in enumerate(cd.inputs):
                if any(isinstance(r, Read) and r.tensor == tns for e in stmt_exprs(region) for r in walk(e)):
                    if tns.dtype not in ("f32", "f16", "q8_0", "q4_0"):
                        continue
                    region, tile = stage(region, tns, "shared", f"t{q}", {TX: wx, TY: wy},
                                         vec=kn.get("vec", 1), pad=kn.get("pad", 0))
                    tiles[tns.name] = tile
            if kn.get("reg") and tiles:
                # register tiles: stage each shared tile inside the ki loop
                region = _stage_regs(region, list(tiles.values()))
        b.emit(region)
    with b.loop("ii", TM, "unroll") as ii:
        with b.loop("jj", TN, "unroll") as jj:
            i, j = I(ii), J(jj)
            with b.if_(land(i < const(M), j < const(N))):
                _store(b, cd, [i, j], subst(cd.body, {ACC: Read(accT, (ii, jj))}))
    return _kernel(cd, b.build(), (wx, wy, 1), (cdiv(N, BN), cdiv(M, BM), 1), kn)


def _stage_regs(region: Stmt, tiles: list) -> Stmt:
    """Inside every ``ki`` loop body, copy the shared-tile operands into registers."""
    def fs(s):
        if isinstance(s, For) and s.var.name.startswith("ki"):
            body = s.body
            for q, tl in enumerate(tiles):
                try:
                    body, _ = stage(body, tl, "local", f"rg{q}")
                except ScheduleError:
                    pass
            return For(s.var, s.extent, body, s.kind)
        return None
    # the ki loop may have been flattened away when BK == 1: stage the whole region then
    out = map_stmt(region, fs=fs)
    return out


# ----------------------------------------------------------------------------- dispatch

def template_for(cd: ComputeDef) -> str:
    if cd.kind == "matmul":
        return "matvec" if cd.meta["M"] == 1 else "matmul"
    if cd.kind == "stencil":
        return "stencil"
    if cd.kind == "row":
        return "row"
    return "injective"


def default_knobs(tmpl: str) -> dict:
    """The untuned schedule: what a straightforward compiler would emit."""
    return {
        "injective": {"wg": 64, "vec": 1, "ept": 1},
        "stencil": {"wx": 64, "wy": 1, "wz": 1, "cz": 1, "stage": 0},
        "row": {"wg": 64, "xlane": "shared", "stage": 0},
        "matvec": {"R": 64, "T": 1, "U": 1, "V": 1, "xlane": "shared", "stage_x": 0, "unroll": 0},
        "matmul": {"BM": 16, "BN": 16, "BK": 1, "TM": 1, "TN": 1, "stage": 0, "vec": 1, "reg": 0,
                   "pad": 0, "unroll": 0},
    }[tmpl]


def space(tmpl: str, cd: ComputeDef, t: Target) -> list[dict]:
    out = []
    if tmpl == "injective":
        for wg, vec, ept in itertools.product((64, 128, 256), (1, 4), (1, 2, 4)):
            out.append({"wg": wg, "vec": vec, "ept": ept})
    elif tmpl == "stencil":
        for wx, wy, wz, cz, st in itertools.product((8, 16, 32, 64), (1, 2, 4, 8), (1, 2, 4), (1, 2), (0, 1)):
            if 32 <= wx * wy * wz <= min(256, t.max_threads):
                out.append({"wx": wx, "wy": wy, "wz": wz, "cz": cz, "stage": st})
    elif tmpl == "row":
        modes = ("shared", "subgroup") if t.subgroup_size else ("shared",)
        for wg, mode, st in itertools.product((64, 128, 256, 512, 1024), modes, (0, 1)):
            if wg <= t.max_threads:
                out.append({"wg": wg, "xlane": mode, "stage": st})
    elif tmpl == "matvec":
        flat = cd.meta["form"] != "blocked"
        modes = ("shared", "subgroup") if t.subgroup_size else ("shared",)
        for R, T, U, V, mode, sx in itertools.product((1, 2, 4, 8, 16, 32, 64, 128), (1, 2, 4, 8, 16, 32, 64, 128, 256), (1, 2, 4),
                                                      (1, 4) if flat else (1,), modes, (0, 1)):
            if R * T > min(256, t.max_threads) or R * T < 64:
                continue
            if mode == "subgroup" and T != t.subgroup_size:
                continue
            if sx and R == 1:
                continue  # staging x only pays off when several rows share it
            out.append({"R": R, "T": T, "U": U, "V": V, "xlane": mode, "stage_x": sx, "unroll": 1})
    elif tmpl == "matmul":
        for BM, BN, BK, TM, TN, vec, reg, pad in itertools.product(
                (32, 64, 128), (32, 64, 128), (8, 16, 32), (2, 4, 8), (2, 4, 8), (1, 4), (0, 1), (0, 1)):
            thr = (BM // TM) * (BN // TN)
            if not (64 <= thr <= min(256, t.max_threads)):
                continue
            smem = 4 * (BM * (BK + pad) + BN * (BK + pad))
            if smem > t.max_shared:
                continue
            out.append({"BM": BM, "BN": BN, "BK": BK, "TM": TM, "TN": TN, "stage": 1, "vec": vec,
                        "reg": reg, "pad": pad, "unroll": 1})
    return out


def shared_bytes(body: Stmt) -> int:
    tiles = set()

    def go(s):
        if isinstance(s, Seq):
            for c in s.body:
                go(c)
        elif isinstance(s, For):
            go(s.body)
        elif isinstance(s, If):
            go(s.then)
            if s.els is not None:
                go(s.els)
        elif isinstance(s, TStore) and s.tensor.scope == "shared":
            tiles.add(s.tensor)
    go(body)
    return sum(4 * t.size for t in tiles)


def schedule(cd: ComputeDef, knobs: dict | None, t: Target) -> Kernel:
    if not t.gpu:
        return cpu_schedule(cd)
    tmpl = template_for(cd)
    kn = dict(default_knobs(tmpl))
    kn.update(knobs or {})
    fn = {"injective": injective_schedule, "stencil": stencil_schedule, "row": row_schedule,
          "matvec": matvec_schedule, "matmul": matmul_schedule}[tmpl]
    k = fn(cd, kn, t)
    k.meta["template"] = tmpl
    smem = shared_bytes(k.body)
    if smem > t.max_shared:
        raise ScheduleError(f"{smem} bytes of shared memory > {t.max_shared}")
    if k.threads > t.max_threads:
        raise ScheduleError(f"{k.threads} threads > {t.max_threads}")
    return k
