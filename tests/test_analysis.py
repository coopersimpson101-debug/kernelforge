"""Index-arithmetic simplifier: randomized equivalence checks + the rewrites the passes rely on."""
import random

from kernelforge.analysis import affine, multiple_of, narrow, rng, simplify_expr
from kernelforge.ir import Bin, Const, Var, const


def ev(e, env):
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Var):
        return env[e]
    if isinstance(e, Bin):
        a, b = ev(e.a, env), ev(e.b, env)
        if e.op == "+": return a + b
        if e.op == "-": return a - b
        if e.op == "*": return a * b
        if e.op == "/": return int(a / b) if b else 0
        if e.op == "%": return a - int(a / b) * b
        if e.op == "min": return min(a, b)
        if e.op == "max": return max(a, b)
        if e.op == "<": return a < b
    raise NotImplementedError(e)


V = [Var("a"), Var("b"), Var("c")]
R = {V[0]: (0, 7), V[1]: (0, 31), V[2]: (0, 100)}


def rand_expr(rnd, d=0):
    if d > 3 or rnd.random() < 0.3:
        return rnd.choice(V) if rnd.random() < 0.6 else const(rnd.randint(0, 40))
    op = rnd.choice(["+", "*", "/", "%", "+", "*", "-"])
    a, b = rand_expr(rnd, d + 1), rand_expr(rnd, d + 1)
    if op in ("/", "%"):
        b = const(rnd.choice([2, 4, 8, 16, 32, 34, 3]))
    if op == "*":
        b = const(rnd.randint(1, 40))
    return Bin(op, a, b, "i32")


def test_simplify_preserves_value():
    rnd = random.Random(0)
    for _ in range(400):
        e = rand_expr(rnd)
        s = simplify_expr(e, R)
        for _ in range(20):
            env = {v: rnd.randint(lo, hi) for v, (lo, hi) in R.items()}
            assert ev(e, env) == ev(s, env), (e, s, env)


def test_divmod_peeling():
    a, b = V[0], V[1]
    # (32*a + b) / 32 == a and % 32 == b when 0 <= b < 32
    assert simplify_expr((a * const(32) + b) / const(32), R) == a
    assert simplify_expr((a * const(32) + b) % const(32), R) == b
    # vector lanes: (4*c + v) % 96 -> (4*c) % 96 + v
    v = Var("v")
    env = dict(R)
    env[v] = (0, 3)
    s = simplify_expr((V[2] * const(4) + v) % const(96), env)
    assert affine(s)[0].get(v) == 1


def test_ranges_and_multiples():
    a, b = V[0], V[1]
    assert rng(a * const(4) + b, R) == (0, 59)
    assert multiple_of(a * const(8) + b * const(4)) == 4
    assert narrow(R, (a * const(4) + b) < const(8))[a] == (0, 1)
