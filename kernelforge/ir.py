"""Expression and loop-nest IR shared by every stage below the tensor graph.

Two layers live here:

* **Expressions** (immutable, hashable, structurally compared): constants, variables, arithmetic,
  intrinsic calls, ``Select``, casts, tensor ``Read`` s (multi-dimensional, before lowering),
  ``Reduce`` (only inside compute definitions) and flat memory loads (after lowering).
* **Statements** (the loop nest): ``For`` / ``Seq`` / ``If`` / ``Store`` / ``Decl`` / ``Assign`` /
  ``Barrier`` plus a ``Kernel`` container that records buffers, workgroup size and grid.

Index arithmetic is always ``i32``; values are ``f32`` (``f32x4`` etc. after vectorisation).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable

# ----------------------------------------------------------------------------- dtypes

SCALAR = ("f32", "i32", "u32", "bool")


def lanes(dt: str) -> int:
    return int(dt.split("x")[1]) if "x" in dt else 1


def elem(dt: str) -> str:
    return dt.split("x")[0]


def vec(dt: str, n: int) -> str:
    return dt if n == 1 else f"{elem(dt)}x{n}"


# ----------------------------------------------------------------------------- expressions

_ids = itertools.count()


class Expr:
    dtype: str = "f32"

    # operator sugar -------------------------------------------------------
    def __add__(self, o): return binop("+", self, o)
    def __radd__(self, o): return binop("+", o, self)
    def __sub__(self, o): return binop("-", self, o)
    def __rsub__(self, o): return binop("-", o, self)
    def __mul__(self, o): return binop("*", self, o)
    def __rmul__(self, o): return binop("*", o, self)
    def __truediv__(self, o): return binop("/", self, o)
    def __rtruediv__(self, o): return binop("/", o, self)
    def __floordiv__(self, o): return binop("/", self, o)
    def __mod__(self, o): return binop("%", self, o)
    def __neg__(self): return binop("-", const(0, self.dtype), self)
    def __lt__(self, o): return binop("<", self, o)
    def __le__(self, o): return binop("<=", self, o)
    def __gt__(self, o): return binop(">", self, o)
    def __ge__(self, o): return binop(">=", self, o)
    def __and__(self, o): return binop("&", self, o)
    def __or__(self, o): return binop("|", self, o)
    def __rshift__(self, o): return binop(">>", self, o)
    def __lshift__(self, o): return binop("<<", self, o)
    def eq(self, o): return binop("==", self, o)
    def ne(self, o): return binop("!=", self, o)


@dataclass(frozen=True, eq=True)
class Const(Expr):
    value: float | int | bool
    dtype: str = "i32"


@dataclass(frozen=True, eq=True)
class Var(Expr):
    name: str
    dtype: str = "i32"


@dataclass(frozen=True, eq=True)
class Bin(Expr):
    op: str
    a: Expr
    b: Expr
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Call(Expr):
    fn: str
    args: tuple
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Select(Expr):
    cond: Expr
    t: Expr
    f: Expr
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Cast(Expr):
    e: Expr
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class TensorRef:
    """A named n-d array: a kernel input/output, or a staged tile (scope shared/local)."""
    name: str
    shape: tuple
    dtype: str  # storage dtype: f32 f16 u32 i32 q8_0 q4_0
    scope: str = "global"

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, eq=True)
class Read(Expr):
    """Multi-dimensional tensor read. ``part`` selects the raw quant value ('q') or block scale
    ('d') of a Q8_0/Q4_0 tensor; '' means the (dequantised) element."""
    tensor: TensorRef
    idx: tuple
    part: str = ""
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Reduce(Expr):
    op: str  # 'sum' | 'max'
    body: Expr
    axes: tuple  # tuple[(Var, extent)]
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Buffer:
    name: str
    dtype: str  # f32 u32 i32 or 'bytes'
    size: int  # elements (bytes for 'bytes')
    scope: str = "global"  # global | shared | local
    readonly: bool = True


@dataclass(frozen=True, eq=True)
class Load(Expr):
    buf: Buffer
    idx: Expr
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class VecLoad(Expr):
    buf: Buffer
    idx: Expr  # index of lane 0, in elements
    n: int = 4
    dtype: str = "f32x4"


@dataclass(frozen=True, eq=True)
class ByteLoad(Expr):
    buf: Buffer
    off: Expr
    dtype: str = "u32"


@dataclass(frozen=True, eq=True)
class WordLoad(Expr):
    """4 bytes starting at an arbitrary (possibly unaligned) byte offset, little endian."""
    buf: Buffer
    off: Expr
    dtype: str = "u32"


@dataclass(frozen=True, eq=True)
class HalfLoad(Expr):
    """IEEE half at an even byte offset, widened to f32."""
    buf: Buffer
    off: Expr
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Lane(Expr):
    """Component ``i`` of a vector expression."""
    e: Expr
    i: int
    dtype: str = "f32"


@dataclass(frozen=True, eq=True)
class Broadcast(Expr):
    e: Expr
    n: int
    dtype: str = "f32x4"


def _cache_hash(cls):
    h = cls.__hash__

    def __hash__(self):
        d = self.__dict__
        if "_h" not in d:
            object.__setattr__(self, "_h", h(self))
        return d["_h"]
    cls.__hash__ = __hash__


for _c in (Const, Var, Bin, Call, Select, Cast, Read, Reduce, Load, VecLoad, ByteLoad, WordLoad, HalfLoad,
           Lane, Broadcast):
    _cache_hash(_c)


def const(v, dtype: str | None = None) -> Const:
    if dtype is None:
        dtype = "bool" if isinstance(v, bool) else ("i32" if isinstance(v, int) else "f32")
    dtype = elem(dtype)
    if dtype == "f32":
        v = float(v)
    elif dtype in ("i32", "u32"):
        v = int(v)
    return Const(v, dtype)


def wrap(x, like: str = "i32") -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return const(x, "bool")
    if isinstance(x, float):
        return const(x, "f32")
    return const(x, like)


CMP = ("<", "<=", ">", ">=", "==", "!=")
LOGIC = ("&&", "||")


def _fold(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/":
        if isinstance(a, float) or isinstance(b, float):
            return a / b
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q  # C-style truncation
    if op == "%":
        if isinstance(a, float):
            return math.fmod(a, b)
        return a - _fold("/", a, b) * b
    if op == "min": return min(a, b)
    if op == "max": return max(a, b)
    if op == "<": return a < b
    if op == "<=": return a <= b
    if op == ">": return a > b
    if op == ">=": return a >= b
    if op == "==": return a == b
    if op == "!=": return a != b
    if op == "&&": return bool(a) and bool(b)
    if op == "||": return bool(a) or bool(b)
    if op == "&": return a & b
    if op == "|": return a | b
    if op == "^": return a ^ b
    if op == ">>": return a >> b
    if op == "<<": return (a << b) & 0xFFFFFFFF if isinstance(a, int) else a
    raise ValueError(op)


def binop(op: str, a, b) -> Expr:
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        raise TypeError("binop needs at least one Expr")
    if not isinstance(a, Expr):
        a = wrap(a, elem(b.dtype))
    if not isinstance(b, Expr):
        b = wrap(b, elem(a.dtype))
    if isinstance(a, Const) and a.dtype in ("i32", "u32") and elem(b.dtype) == "f32":
        a = const(a.value, "f32")
    if isinstance(b, Const) and b.dtype in ("i32", "u32") and elem(a.dtype) == "f32":
        b = const(b.value, "f32")
    if isinstance(a, Const) and isinstance(b, Const):
        v = _fold(op, a.value, b.value)
        return const(v, "bool" if op in CMP + LOGIC else a.dtype)
    dt = "bool" if op in CMP + LOGIC else (a.dtype if lanes(a.dtype) >= lanes(b.dtype) else b.dtype)
    isint = elem(dt) in ("i32", "u32")
    # cheap algebraic identities (exact for both ints and floats)
    if op == "+":
        if isinstance(b, Const) and b.value == 0: return a
        if isinstance(a, Const) and a.value == 0: return b
    if op == "-" and isinstance(b, Const) and b.value == 0: return a
    if op == "*":
        if isinstance(b, Const) and b.value == 1: return a
        if isinstance(a, Const) and a.value == 1: return b
        if isint and ((isinstance(a, Const) and a.value == 0) or (isinstance(b, Const) and b.value == 0)):
            return const(0, dt)
    if op == "/" and isinstance(b, Const) and b.value == 1: return a
    if op in (">>", "<<") and isinstance(b, Const) and b.value == 0: return a
    if op == "%" and isint and isinstance(b, Const) and b.value == 1: return const(0, dt)
    if op == "&&":
        if isinstance(a, Const): return b if a.value else a
        if isinstance(b, Const): return a if b.value else b
    if op == "||":
        if isinstance(a, Const): return a if a.value else b
        if isinstance(b, Const): return b if b.value else a
    return Bin(op, a, b, dt)


def call(fn: str, *args, dtype: str | None = None) -> Expr:
    args = tuple(wrap(a, "f32") for a in args)
    if dtype is None:
        dtype = args[0].dtype if args else "f32"
    return Call(fn, args, dtype)


def select(c: Expr, t, f) -> Expr:
    t = wrap(t, "f32")
    f = wrap(f, t.dtype)
    if isinstance(c, Const):
        return t if c.value else f
    if t == f:
        return t
    return Select(c, t, f, t.dtype)


def cast(e: Expr, dtype: str) -> Expr:
    if e.dtype == dtype:
        return e
    if isinstance(e, Const):
        return const(e.value, dtype)
    return Cast(e, dtype)


def emin(a, b): return binop("min", a, b)
def emax(a, b): return binop("max", a, b)
def land(*cs):
    out = const(True)
    for c in cs:
        out = binop("&&", out, c)
    return out


def fresh(prefix: str, dtype: str = "i32") -> Var:
    return Var(f"{prefix}{next(_ids)}", dtype)


# ----------------------------------------------------------------------------- traversal

def children(e: Expr) -> tuple:
    if isinstance(e, Bin): return (e.a, e.b)
    if isinstance(e, Call): return e.args
    if isinstance(e, Select): return (e.cond, e.t, e.f)
    if isinstance(e, Cast): return (e.e,)
    if isinstance(e, Read): return e.idx
    if isinstance(e, Reduce): return (e.body,)
    if isinstance(e, (Load, VecLoad)): return (e.idx,)
    if isinstance(e, (ByteLoad, WordLoad, HalfLoad)): return (e.off,)
    if isinstance(e, (Lane, Broadcast)): return (e.e,)
    return ()


def rebuild(e: Expr, ch: tuple) -> Expr:
    if isinstance(e, Bin): return binop(e.op, ch[0], ch[1])
    if isinstance(e, Call): return Call(e.fn, tuple(ch), e.dtype)
    if isinstance(e, Select): return select(ch[0], ch[1], ch[2])
    if isinstance(e, Cast): return cast(ch[0], e.dtype)
    if isinstance(e, Read): return Read(e.tensor, tuple(ch), e.part, e.dtype)
    if isinstance(e, Reduce): return Reduce(e.op, ch[0], e.axes, e.dtype)
    if isinstance(e, Load): return Load(e.buf, ch[0], e.dtype)
    if isinstance(e, VecLoad): return VecLoad(e.buf, ch[0], e.n, e.dtype)
    if isinstance(e, ByteLoad): return ByteLoad(e.buf, ch[0], e.dtype)
    if isinstance(e, WordLoad): return WordLoad(e.buf, ch[0], e.dtype)
    if isinstance(e, HalfLoad): return HalfLoad(e.buf, ch[0], e.dtype)
    if isinstance(e, Lane): return Lane(ch[0], e.i, e.dtype)
    if isinstance(e, Broadcast): return Broadcast(ch[0], e.n, e.dtype)
    return e


def transform(e: Expr, fn) -> Expr:
    """Post-order rewrite: ``fn(node_with_rewritten_children) -> Expr | None``."""
    memo: dict = {}

    def go(x):
        if x in memo:
            return memo[x]
        ch = children(x)
        y = rebuild(x, tuple(go(c) for c in ch)) if ch else x
        r = fn(y)
        out = y if r is None else r
        memo[x] = out
        return out

    return go(e)


def subst(e: Expr, m: dict) -> Expr:
    if not m:
        return e
    return transform(e, lambda x: m.get(x) if isinstance(x, Var) else None)


def walk(e: Expr) -> Iterable[Expr]:
    stack = [e]
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        yield x
        stack.extend(children(x))


def free_vars(e: Expr) -> set:
    return {x for x in walk(e) if isinstance(x, Var)}


# ----------------------------------------------------------------------------- statements

class Stmt:
    pass


@dataclass
class Seq(Stmt):
    body: list = field(default_factory=list)


@dataclass
class For(Stmt):
    var: Var
    extent: int
    body: Stmt
    kind: str = "serial"  # serial | unroll | vectorize


@dataclass
class If(Stmt):
    cond: Expr
    then: Stmt
    els: Stmt | None = None


@dataclass
class Store(Stmt):
    buf: Buffer
    idx: Expr
    value: Expr


@dataclass
class Decl(Stmt):
    """Declare a scalar (or vector) local; ``mutable`` locals may be reassigned."""
    var: Var
    init: Expr
    mutable: bool = False


@dataclass
class Assign(Stmt):
    var: Var
    value: Expr


@dataclass
class Barrier(Stmt):
    pass


@dataclass
class Comment(Stmt):
    text: str


@dataclass
class TStore(Stmt):
    """Tensor-level store (before lowering): out[idx...] = value."""
    tensor: TensorRef
    idx: tuple
    value: Expr


THREAD = {"tx": 0, "ty": 1, "tz": 2}
BLOCK = {"bx": 0, "by": 1, "bz": 2}
TX, TY, TZ = Var("tx"), Var("ty"), Var("tz")
BX, BY, BZ = Var("bx"), Var("by"), Var("bz")


@dataclass
class Kernel:
    name: str
    params: list  # list[TensorRef] in binding order (inputs then output)
    outputs: list  # names of written params
    body: Stmt
    workgroup: tuple = (64, 1, 1)
    grid: tuple = (1, 1, 1)
    allocs: list = field(default_factory=list)  # shared / local Buffers
    features: set = field(default_factory=set)  # e.g. {'subgroups'}
    meta: dict = field(default_factory=dict)
    buffers: dict = field(default_factory=dict)  # tensor name -> Buffer (after lowering)
    alias: dict = field(default_factory=dict)  # output name -> input name it overwrites in place

    @property
    def threads(self) -> int:
        return math.prod(self.workgroup)


def map_stmt(s: Stmt, fs=None, fe=None) -> Stmt:
    """Rebuild a statement tree. ``fe`` rewrites every expression; ``fs`` (post-order) may replace a
    statement (return None to keep it)."""
    def E(x):
        return fe(x) if fe is not None and isinstance(x, Expr) else x

    def go(t: Stmt) -> Stmt:
        if isinstance(t, Seq):
            r = Seq([go(c) for c in t.body])
        elif isinstance(t, For):
            r = For(t.var, t.extent, go(t.body), t.kind)
        elif isinstance(t, If):
            r = If(E(t.cond), go(t.then), go(t.els) if t.els is not None else None)
        elif isinstance(t, Store):
            r = Store(t.buf, E(t.idx), E(t.value))
        elif isinstance(t, TStore):
            r = TStore(t.tensor, tuple(E(i) for i in t.idx), E(t.value))
        elif isinstance(t, Decl):
            r = Decl(t.var, E(t.init), t.mutable)
        elif isinstance(t, Assign):
            r = Assign(t.var, E(t.value))
        else:
            r = t
        if fs is not None:
            y = fs(r)
            if y is not None:
                return y
        return r

    return go(s)


def stmt_exprs(s: Stmt) -> Iterable[Expr]:
    if isinstance(s, Seq):
        for c in s.body:
            yield from stmt_exprs(c)
    elif isinstance(s, For):
        yield from stmt_exprs(s.body)
    elif isinstance(s, If):
        yield s.cond
        yield from stmt_exprs(s.then)
        if s.els is not None:
            yield from stmt_exprs(s.els)
    elif isinstance(s, Store):
        yield s.idx
        yield s.value
    elif isinstance(s, TStore):
        yield from s.idx
        yield s.value
    elif isinstance(s, Decl):
        yield s.init
    elif isinstance(s, Assign):
        yield s.value


def flatten_seq(s: Stmt) -> Stmt:
    def fs(t):
        if isinstance(t, Seq):
            out = []
            for c in t.body:
                if isinstance(c, Seq):
                    out.extend(c.body)
                elif c is not None:
                    out.append(c)
            return Seq(out)
        return None
    return map_stmt(s, fs=fs)


# ----------------------------------------------------------------------------- builder

class Builder:
    """Tiny imperative helper used by schedule templates to emit loop nests."""

    def __init__(self):
        self.stack: list[list] = [[]]

    def emit(self, s: Stmt):
        self.stack[-1].append(s)
        return s

    def push(self):
        self.stack.append([])

    def pop(self) -> Seq:
        return Seq(self.stack.pop())

    def loop(self, name: str, extent: int, kind: str = "serial"):
        b = self
        v = fresh(name)

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

    def if_(self, cond: Expr):
        b = self

        class _I:
            def __enter__(self_):
                b.push()

            def __exit__(self_, *a):
                body = b.pop()
                if isinstance(cond, Const):
                    if cond.value:
                        b.emit(body)
                else:
                    b.emit(If(cond, body))
        return _I()

    def let(self, prefix: str, init: Expr) -> Expr:
        if isinstance(init, (Const, Var)):
            return init
        v = fresh(prefix, init.dtype)
        self.emit(Decl(v, init, False))
        return v

    def var(self, prefix: str, init: Expr) -> Var:
        v = fresh(prefix, init.dtype)
        self.emit(Decl(v, init, True))
        return v

    def assign(self, v: Var, e: Expr):
        self.emit(Assign(v, e))

    def barrier(self):
        self.emit(Barrier())

    def build(self) -> Stmt:
        assert len(self.stack) == 1
        return flatten_seq(Seq(self.stack[0]))


# ----------------------------------------------------------------------------- printer

def show_expr(e: Expr) -> str:
    if isinstance(e, Const):
        if e.dtype == "f32":
            return repr(float(e.value))
        return str(e.value).lower() if e.dtype == "bool" else str(e.value)
    if isinstance(e, Var): return e.name
    if isinstance(e, Bin):
        if e.op in ("min", "max"):
            return f"{e.op}({show_expr(e.a)}, {show_expr(e.b)})"
        return f"({show_expr(e.a)} {e.op} {show_expr(e.b)})"
    if isinstance(e, Call): return f"{e.fn}({', '.join(show_expr(a) for a in e.args)})"
    if isinstance(e, Select): return f"select({show_expr(e.cond)}, {show_expr(e.t)}, {show_expr(e.f)})"
    if isinstance(e, Cast): return f"{e.dtype}({show_expr(e.e)})"
    if isinstance(e, Read):
        p = f".{e.part}" if e.part else ""
        return f"{e.tensor.name}{p}[{', '.join(show_expr(i) for i in e.idx)}]"
    if isinstance(e, Reduce):
        ax = ", ".join(f"{v.name}<{n}" for v, n in e.axes)
        return f"{e.op}({show_expr(e.body)} for {ax})"
    if isinstance(e, Load): return f"{e.buf.name}[{show_expr(e.idx)}]"
    if isinstance(e, VecLoad): return f"{e.buf.name}[{show_expr(e.idx)} : +{e.n}]"
    if isinstance(e, ByteLoad): return f"u8({e.buf.name}, {show_expr(e.off)})"
    if isinstance(e, WordLoad): return f"u32le({e.buf.name}, {show_expr(e.off)})"
    if isinstance(e, HalfLoad): return f"f16({e.buf.name}, {show_expr(e.off)})"
    if isinstance(e, Lane): return f"{show_expr(e.e)}.{'xyzw'[e.i]}"
    if isinstance(e, Broadcast): return f"splat{e.n}({show_expr(e.e)})"
    return repr(e)


def show_stmt(s: Stmt, ind: int = 0) -> str:
    p = "  " * ind
    if isinstance(s, Seq):
        return "\n".join(show_stmt(c, ind) for c in s.body)
    if isinstance(s, For):
        tag = "" if s.kind == "serial" else f"  # {s.kind}"
        return f"{p}for {s.var.name} in 0..{s.extent}:{tag}\n{show_stmt(s.body, ind + 1)}"
    if isinstance(s, If):
        out = f"{p}if {show_expr(s.cond)}:\n{show_stmt(s.then, ind + 1)}"
        if s.els is not None:
            out += f"\n{p}else:\n{show_stmt(s.els, ind + 1)}"
        return out
    if isinstance(s, Store): return f"{p}{s.buf.name}[{show_expr(s.idx)}] = {show_expr(s.value)}"
    if isinstance(s, TStore):
        return f"{p}{s.tensor.name}[{', '.join(show_expr(i) for i in s.idx)}] = {show_expr(s.value)}"
    if isinstance(s, Decl): return f"{p}{'var' if s.mutable else 'let'} {s.var.name} = {show_expr(s.init)}"
    if isinstance(s, Assign): return f"{p}{s.var.name} = {show_expr(s.value)}"
    if isinstance(s, Barrier): return f"{p}barrier()"
    if isinstance(s, Comment): return f"{p}# {s.text}"
    return f"{p}{s!r}"
