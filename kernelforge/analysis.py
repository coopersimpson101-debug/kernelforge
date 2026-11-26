"""Integer analyses used by the passes: affine forms, interval ranges, divisibility and an
algebraic simplifier for index arithmetic (the part that makes unrolled/tiled index maths collapse).

All integer division here is C/WGSL/MSL truncating division. The rewrites below only fire when the
operands are proven non-negative, where truncation equals floor division.
"""
from __future__ import annotations

import math

from .ir import (Bin, ByteLoad, Call, Cast, Const, Decl, Expr, For, If, Kernel, Seq, Select, Stmt, Var,
                 binop, children, const, elem, map_stmt, rebuild, select, show_expr, BLOCK, THREAD)

Range = tuple  # (lo, hi) inclusive, ints


# ----------------------------------------------------------------------------- affine forms

def affine(e: Expr):
    """Return (terms, const) with terms: {atom: coef} such that e == sum(coef*atom) + const.
    Atoms are Vars or any non-affine integer sub-expression. None for non-integer expressions."""
    if isinstance(e, Const) and e.dtype in ("i32", "u32"):
        return {}, int(e.value)
    if e.dtype != "i32":  # u32 wraps: never re-associate it
        return None
    if isinstance(e, Bin) and e.op in ("+", "-"):
        a, b = affine(e.a), affine(e.b)
        if a is None or b is None:
            return None
        s = 1 if e.op == "+" else -1
        t = dict(a[0])
        for k, v in b[0].items():
            t[k] = t.get(k, 0) + s * v
            if t[k] == 0:
                del t[k]
        return t, a[1] + s * b[1]
    if isinstance(e, Bin) and e.op == "*":
        for x, y in ((e.a, e.b), (e.b, e.a)):
            if isinstance(y, Const):
                a = affine(x)
                if a is None:
                    return None
                c = int(y.value)
                return ({k: v * c for k, v in a[0].items() if v * c}, a[1] * c)
    return {e: 1}, 0


def _key(e):
    return show_expr(e)


def from_affine(terms: dict, c: int, dtype: str = "i32") -> Expr:
    out = None
    for atom in sorted(terms, key=_key):
        k = terms[atom]
        t = atom if k == 1 else binop("*", atom, const(k, dtype))
        out = t if out is None else binop("+", out, t)
    if out is None:
        return const(c, dtype)
    if c:
        out = binop("+", out, const(c, dtype))
    return out


# ----------------------------------------------------------------------------- ranges

def _tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def rng(e: Expr, env: dict):
    """Conservative inclusive integer interval of ``e`` or None if unknown. ``env`` maps Vars to
    ranges and, from ``narrow``, frozenset(affine terms) to ranges of that affine sum (facts)."""
    r = _rng(e, env)
    if isinstance(e, Bin) and e.dtype == "i32" and e.op in ("+", "-", "*"):
        af = affine(e)
        if af is not None and af[0]:
            f = env.get(frozenset(af[0].items()))
            if f is not None:
                lo, hi = f[0] + af[1], f[1] + af[1]
                r = (lo, hi) if r is None else (max(r[0], lo), min(r[1], hi))
    return r


def _rng(e: Expr, env: dict):
    if isinstance(e, Const):
        if e.dtype == "bool":
            return (int(e.value), int(e.value))
        return (int(e.value), int(e.value)) if elem(e.dtype) in ("i32", "u32") else None
    if isinstance(e, Var):
        return env.get(e)
    if isinstance(e, ByteLoad):
        return (0, 255)
    if isinstance(e, Cast):
        if e.dtype in ("i32", "u32"):
            return rng(e.e, env)
        return None
    if isinstance(e, Select):
        a, b = rng(e.t, env), rng(e.f, env)
        if a is None or b is None:
            return None
        return (min(a[0], b[0]), max(a[1], b[1]))
    if isinstance(e, Bin):
        if e.op in ("<", "<=", ">", ">=", "==", "!=", "&&", "||"):
            return (0, 1)
        a, b = rng(e.a, env), rng(e.b, env)
        if a is None or b is None:
            return None
        if e.op == "+": return (a[0] + b[0], a[1] + b[1])
        if e.op == "-": return (a[0] - b[1], a[1] - b[0])
        if e.op == "*":
            ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
            return (min(ps), max(ps))
        if e.op == "/" and b[0] == b[1] and b[0] > 0:
            return (_tdiv(a[0], b[0]), _tdiv(a[1], b[0]))
        if e.op == "%" and b[0] == b[1] and b[0] > 0 and a[0] >= 0:
            if a[1] < b[0]:
                return a
            return (0, b[0] - 1)
        if e.op == "min": return (min(a[0], b[0]), min(a[1], b[1]))
        if e.op == "max": return (max(a[0], b[0]), max(a[1], b[1]))
        if e.op == "&" and b[0] == b[1] and b[0] >= 0 and a[0] >= 0:
            return (0, min(a[1], b[0]))
        if e.op == ">>" and b[0] == b[1] and a[0] >= 0:
            return (a[0] >> b[0], a[1] >> b[0])
        if e.op == "<<" and b[0] == b[1] and a[0] >= 0 and (a[1] << b[0]) < 2 ** 31:
            return (a[0] << b[0], a[1] << b[0])
        return None
    return None


def multiple_of(e: Expr, defs: dict | None = None) -> int:
    """Largest known m such that e is always a multiple of m (1 if unknown). ``defs`` maps
    immutable let-variables to their definitions."""
    if isinstance(e, Const) and elem(e.dtype) in ("i32", "u32"):
        return abs(int(e.value)) or 1 << 30
    if isinstance(e, Var) and defs and e in defs:
        return multiple_of(defs[e], defs)
    if isinstance(e, Bin):
        if e.op in ("+", "-"):
            return math.gcd(multiple_of(e.a, defs), multiple_of(e.b, defs))
        if e.op == "*":
            return min(multiple_of(e.a, defs) * multiple_of(e.b, defs), 1 << 30)
        if e.op == "%" and isinstance(e.b, Const):
            return math.gcd(multiple_of(e.a, defs), int(e.b.value))
        if e.op == "<<" and isinstance(e.b, Const):
            return min(multiple_of(e.a, defs) << int(e.b.value), 1 << 30)
    return 1


def let_defs(s: Stmt) -> dict:
    out = {}

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
        elif isinstance(t, Decl) and not t.mutable:
            out[t.var] = t.init
    go(s)
    return out


# ----------------------------------------------------------------------------- simplifier

def _divisors_desc(c: int):
    return [d for d in range(c, 0, -1) if c % d == 0]


def _split_divmod(op: str, a: Expr, c: int, env: dict):
    af = affine(a)
    if af is None:
        return None
    terms, k = af
    whole = rng(a, env)
    if whole is not None and whole[0] >= 0 and whole[1] < c:
        return const(0, a.dtype) if op == "/" else a
    for g in _divisors_desc(c):
        if g == 1:
            break
        P = {t: v for t, v in terms.items() if v % g == 0}
        R = {t: v for t, v in terms.items() if v % g != 0}
        kp, kr = (k // g) * g, k % g
        pe, re_ = from_affine(P, kp, a.dtype), from_affine(R, kr, a.dtype)
        rp, rr = rng(pe, env), rng(re_, env)
        if rp is None or rr is None or rp[0] < 0 or rr[0] < 0 or rr[1] >= g:
            continue
        if g == c:
            if op == "%":
                return re_
            return from_affine({t: v // c for t, v in P.items()}, kp // c, a.dtype)
        if not R and kr == 0:
            continue  # nothing to peel off
        if op == "%":
            return binop("+", binop("%", pe, const(c, a.dtype)), re_)
        return binop("/", pe, const(c, a.dtype))
    # peel exact multiples of c even when the remainder may exceed c:  (c*X + R)/c = X + R/c
    P = {t: v for t, v in terms.items() if v % c == 0}
    kp, kr = (k // c) * c, k % c
    if P or kp:
        R = {t: v for t, v in terms.items() if v % c != 0}
        pe, re_ = from_affine(P, kp, a.dtype), from_affine(R, kr, a.dtype)
        rp, rr = rng(pe, env), rng(re_, env)
        if rp is not None and rr is not None and rp[0] >= 0 and rr[0] >= 0 and R:
            X = from_affine({t: v // c for t, v in P.items()}, kp // c, a.dtype)
            if op == "/":
                return binop("+", X, binop("/", re_, const(c, a.dtype)))
            return binop("%", re_, const(c, a.dtype))
    return None


def simplify_expr(e: Expr, env: dict) -> Expr:
    memo: dict = {}

    def go(x: Expr) -> Expr:
        if x in memo:
            return memo[x]
        if isinstance(x, Select):
            c = go(x.cond)
            if isinstance(c, Const):
                y = go(x.t) if c.value else go(x.f)
            else:
                t = simplify_expr(x.t, narrow(env, c))
                y = step(select(c, t, go(x.f)))
            memo[x] = y
            return y
        ch = children(x)
        y = rebuild(x, tuple(go(c) for c in ch)) if ch else x
        y = step(y)
        memo[x] = y
        return y

    def step(y: Expr) -> Expr:
        if isinstance(y, Bin):
            isint = elem(y.dtype) in ("i32", "u32") and "x" not in y.dtype
            if y.op in ("/", "%") and isint and isinstance(y.b, Const) and int(y.b.value) > 0:
                r = _split_divmod(y.op, y.a, int(y.b.value), env)
                if r is not None:
                    return go(r) if r != y else r
            if y.op in ("<", "<=", ">", ">=", "==", "!="):
                a, b = rng(y.a, env), rng(y.b, env)
                if a is not None and b is not None:
                    if y.op == "<" and a[1] < b[0] or y.op == "<=" and a[1] <= b[0] \
                            or y.op == ">" and a[0] > b[1] or y.op == ">=" and a[0] >= b[1]:
                        return const(True)
                    if y.op == "<" and a[0] >= b[1] or y.op == "<=" and a[0] > b[1] \
                            or y.op == ">" and a[1] <= b[0] or y.op == ">=" and a[1] < b[0]:
                        return const(False)
                    if y.op == "==" and a[0] == a[1] == b[0] == b[1]:
                        return const(True)
                    if y.op in ("==", "!=") and (a[1] < b[0] or b[1] < a[0]):
                        return const(y.op == "!=")
            if y.op in ("min", "max") and isint:
                a, b = rng(y.a, env), rng(y.b, env)
                if a is not None and b is not None:
                    if a[1] <= b[0]:
                        return y.a if y.op == "min" else y.b
                    if b[1] <= a[0]:
                        return y.b if y.op == "min" else y.a
            if isint and y.op in ("+", "-", "*"):
                af = affine(y)
                if af is not None:
                    return from_affine(af[0], af[1], y.dtype)
        if isinstance(y, Select) and isinstance(y.cond, Const):
            return y.t if y.cond.value else y.f
        return y

    return go(e)


def narrow(env: dict, cond: Expr) -> dict:
    """Ranges that hold inside ``if (cond)``: uses conjuncts of the form affine < c / <= c / >= c."""
    out = dict(env)
    terms = []

    def conj(c):
        if isinstance(c, Bin) and c.op == "&&":
            conj(c.a)
            conj(c.b)
        else:
            terms.append(c)
    conj(cond)
    for _ in range(2):
        for c in terms:
            if not isinstance(c, Bin) or c.op not in ("<", "<=", ">=", ">"):
                continue
            a, b = (c.a, c.b)
            if not isinstance(b, Const) or elem(b.dtype) not in ("i32", "u32"):
                continue
            af = affine(a)
            if af is None:
                continue
            t, k0 = af
            bound = int(b.value)
            if len(t) > 1 or (t and next(iter(t.values())) != 1):
                key = frozenset(t.items())
                lo, hi = out.get(key, (-(1 << 40), 1 << 40))
                if c.op == "<": hi = min(hi, bound - 1 - k0)
                elif c.op == "<=": hi = min(hi, bound - k0)
                elif c.op == ">=": lo = max(lo, bound - k0)
                elif c.op == ">": lo = max(lo, bound + 1 - k0)
                out[key] = (lo, hi)
            for v, coef in t.items():
                if not isinstance(v, Var) or v not in out:
                    continue
                rest = from_affine({x: y for x, y in t.items() if x != v}, k0)
                rr = rng(rest, out)
                if rr is None:
                    continue
                lo, hi = out[v]
                if c.op in ("<", "<=") and coef > 0:
                    lim = bound - 1 if c.op == "<" else bound
                    hi = min(hi, (lim - rr[0]) // coef)
                elif c.op in (">=", ">") and coef > 0:
                    lim = bound + 1 if c.op == ">" else bound
                    lo = max(lo, -((rr[1] - lim) // coef))
                if lo <= hi:
                    out[v] = (lo, hi)
    return out


def thread_env(k: Kernel) -> dict:
    env = {}
    for n, i in THREAD.items():
        env[Var(n)] = (0, k.workgroup[i] - 1)
    for n, i in BLOCK.items():
        env[Var(n)] = (0, k.grid[i] - 1)
    return env


def simplify_stmt(s: Stmt, env: dict) -> Stmt:
    """Simplify expressions with the ranges of enclosing loops/lets, prune decided Ifs."""
    def go(t: Stmt, env: dict) -> Stmt | None:
        if isinstance(t, Seq):
            out = []
            env = dict(env)
            for c in t.body:
                r = go(c, env)
                if isinstance(c, Decl) and not c.mutable and isinstance(r, Decl):
                    rr = rng(r.init, env)
                    if rr is not None:
                        env[c.var] = rr
                if r is None:
                    continue
                if isinstance(r, Seq):
                    out.extend(r.body)
                else:
                    out.append(r)
            return Seq(out)
        if isinstance(t, For):
            e2 = dict(env)
            e2[t.var] = (0, t.extent - 1)
            b = go(t.body, e2)
            if b is None or (isinstance(b, Seq) and not b.body):
                return None
            return For(t.var, t.extent, b, t.kind)
        if isinstance(t, If):
            c = simplify_expr(t.cond, env)
            if isinstance(c, Const):
                if c.value:
                    return go(t.then, env)
                return go(t.els, env) if t.els is not None else None
            th = go(t.then, narrow(env, c))
            el = go(t.els, env) if t.els is not None else None
            if th is None and el is None:
                return None
            return If(c, th if th is not None else Seq([]), el)
        return map_stmt(t, fe=lambda x: simplify_expr(x, env))

    r = go(s, env)
    return r if r is not None else Seq([])


def simplify_kernel(k: Kernel) -> Kernel:
    k.body = simplify_stmt(k.body, thread_env(k))
    return k
