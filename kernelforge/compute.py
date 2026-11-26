"""Compute definitions (the "what", TVM ``te.compute`` style) for a fused group.

A ``ComputeDef`` says, for every output index, which value is stored, as an index expression over
``Read`` s of the group's external inputs. Fusion inside a group is plain inlining: a member node's
expression is substituted where it is read. Nothing here decides loop order, tiling or threads:
that is the schedule's job (``schedule.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import quant
from .fusion import Group
from .graph import AIR, FLUID, SOLID, Node
from .ir import (Expr, Read, Reduce, TensorRef, Var, binop, call, cast, const, emax, emin, fresh,
                 land, select)

ACC = Var("acc", "f32")


@dataclass
class ComputeDef:
    name: str
    kind: str  # injective | matmul | row | stencil
    out: TensorRef
    axes: list  # output axis Vars (one per output dim)
    inputs: list  # TensorRefs read by the kernel, binding order
    body: Expr  # stored value (uses ACC for matmul, reduction vars for row)
    reduce: Reduce | None = None  # matmul / matvec
    reductions: list = field(default_factory=list)  # row: [(Var, Reduce)]
    predicate: Expr | None = None  # store only where true (red-black colour)
    alias: dict = field(default_factory=dict)  # out name -> input name (in-place)
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        from .ir import show_expr
        ax = ", ".join(v.name for v in self.axes)
        lines = [f"compute {self.name} [{self.kind}]: {self.out.name}{list(self.out.shape)}"]
        for v, r in self.reductions:
            lines.append(f"  {v.name} = {show_expr(r)}")
        if self.reduce is not None:
            lines.append(f"  acc = {show_expr(self.reduce)}")
        pred = f"  where {show_expr(self.predicate)}" if self.predicate is not None else ""
        lines.append(f"  {self.out.name}[{ax}] = {show_expr(self.body)}{pred}")
        return "\n".join(lines)


def tref(n: Node) -> TensorRef:
    return TensorRef(n.name, tuple(n.shape), n.dtype)


def bcast_idx(shape: tuple, idx: tuple) -> tuple:
    r = len(shape)
    tail = idx[len(idx) - r:] if r else ()
    return tuple(const(0) if s == 1 else i for s, i in zip(shape, tail))


def _unary(op: str, x: Expr) -> Expr:
    if op == "neg": return const(0.0) - x
    if op == "exp": return call("exp", x)
    if op == "silu": return x / (const(1.0) + call("exp", const(0.0) - x))
    if op == "sigmoid": return const(1.0) / (const(1.0) + call("exp", const(0.0) - x))
    if op == "relu": return emax(x, const(0.0))
    if op == "sqrt": return call("sqrt", x)
    if op == "rsqrt": return call("rsqrt", x)
    if op == "tanh": return call("tanh", x)
    if op == "abs": return call("abs", x)
    if op == "gelu":
        inner = const(0.7978845608028654) * (x + const(0.044715) * x * x * x)
        return const(0.5) * x * (const(1.0) + call("tanh", inner))
    raise NotImplementedError(op)


_BIN = {"add": "+", "sub": "-", "mul": "*", "div": "/", "maximum": "max", "minimum": "min"}


class _Lowerer:
    def __init__(self, G: Group, form: str):
        self.G = G
        self.members = set(G.nodes)
        self.inputs: list[TensorRef] = []
        self.form = form  # 'flat' | 'blocked' (quantised linear)
        self.red_vars: dict = {}
        self.reductions: list = []
        self.reduce: Reduce | None = None

    def read(self, n: Node, idx: tuple, part: str = "") -> Expr:
        t = tref(n)
        if t not in self.inputs:
            self.inputs.append(t)
        dt = "u32" if n.dtype == "u32" else "f32"
        return Read(t, bcast_idx(n.shape, idx), part, dt)

    def expr(self, n: Node, idx: tuple) -> Expr:
        if n not in self.members:
            return self.read(n, idx)
        if n.op in _BIN:
            a = self.expr(n.inputs[0], idx)
            b = self.expr(n.inputs[1], idx)
            return binop(_BIN[n.op], a, b)
        if n.op == "adds": return self.expr(n.inputs[0], idx) + const(n.attrs["c"])
        if n.op == "muls": return self.expr(n.inputs[0], idx) * const(n.attrs["c"])
        if n.op in ("neg", "exp", "silu", "sigmoid", "relu", "sqrt", "rsqrt", "tanh", "abs", "gelu"):
            return _unary(n.op, self.expr(n.inputs[0], idx))
        if n.op in ("sum", "max", "mean"):
            if n not in self.red_vars:
                x = n.inputs[0]
                j = fresh("r")
                body = self.expr(x, tuple(idx[:-1]) + (j,))
                red = Reduce("max" if n.op == "max" else "sum", body, ((j, x.shape[-1]),))
                v = Var(f"red{len(self.red_vars)}", "f32")
                self.red_vars[n] = v
                self.reductions.append((v, red))
                if n.op == "mean":
                    mv = Var(f"red{len(self.red_vars)}m", "f32")
                    self.red_vars[n] = mv
                    self.reductions.append((mv, v * const(1.0 / x.shape[-1])))
            return self.red_vars[n]
        if n.op in ("matmul", "linear"):
            assert self.reduce is None
            a, b = n.inputs
            K = a.shape[-1]
            i, j = (idx[0] if len(idx) == 2 else const(0)), idx[-1]
            arow = (lambda kk: (i, kk)) if len(a.shape) == 2 else (lambda kk: (kk,))
            if n.op == "matmul":
                k = fresh("k")
                self.reduce = Reduce("sum", self.expr(a, arow(k)) * self.expr(b, (k, j)), ((k, K),))
            elif b.dtype in quant.BLOCK_BYTES and b not in self.members and self.form == "blocked":
                bb, jj = fresh("b"), fresh("j")
                kk = bb * const(quant.QK) + jj
                inner = Reduce("sum", self.read(b, (j, kk), "q") * self.expr(a, arow(kk)),
                               ((jj, quant.QK),))
                self.reduce = Reduce("sum", self.read(b, (j, bb), "d") * inner, ((bb, K // quant.QK),))
            else:
                k = fresh("k")
                self.reduce = Reduce("sum", self.expr(a, arow(k)) * self.expr(b, (j, k)), ((k, K),))
            return ACC
        if n.op in ("jacobi7", "rbsor"):
            return self.stencil(n, idx)
        raise NotImplementedError(n.op)

    def stencil(self, n: Node, idx: tuple) -> Expr:
        p, rhs, cell = n.inputs
        nz, ny, nx = n.attrs["dims"]
        z, y, x = idx

        def in_cells(c):
            return land(c[0] >= 0, c[1] >= 0, c[2] >= 0, c[0] < nz, c[1] < ny, c[2] < nx)

        def type_at(c):
            return select(in_cells(c), self.expr(cell, c), const(SOLID, "u32"))

        diag, s = const(0.0), const(0.0)
        # same visiting order as flipgpu's neighbourSum: axis x, y, z; sign -1 then +1
        for ax in (2, 1, 0):
            for sgn in (-1, 1):
                c = [z, y, x]
                c[ax] = c[ax] + sgn
                c = tuple(c)
                t = type_at(c)
                diag = diag + select(t.ne(const(SOLID, "u32")), const(1.0), const(0.0))
                s = s + select(t.eq(const(FLUID, "u32")), self.expr(p, c), const(0.0))
        me = type_at((z, y, x))
        gs = select(diag > const(0.0), (self.expr(rhs, idx) + s) / diag, const(0.0))
        if n.op == "jacobi7":
            return select(me.ne(const(FLUID, "u32")), const(0.0), gs)
        w = n.attrs["omega"]
        par = n.attrs["parity"]
        self.predicate = land(((x + y + z) & const(1)).eq(const(par)), me.eq(const(FLUID, "u32")),
                              diag.ne(const(0.0)))
        return const(1.0 - w) * self.expr(p, idx) + const(w) * ((self.expr(rhs, idx) + s) / diag)


def compute_def(G: Group, form: str = "flat") -> ComputeDef:
    out = G.output
    L = _Lowerer(G, form)
    L.predicate = None
    if G.kind == "row":
        rows = G.rows
        axes = [fresh("i") for _ in out.shape]
    else:
        axes = [fresh("i") for _ in out.shape]
    body = L.expr(out, tuple(axes))
    cd = ComputeDef(name=f"k_{out.name}", kind=G.kind, out=tref(out), axes=axes, inputs=L.inputs,
                    body=body, reduce=L.reduce, reductions=L.reductions, predicate=L.predicate)
    if G.kind == "row":
        cd.meta["rows"] = rows
    if G.anchor is not None and G.anchor.op == "rbsor":
        cd.alias = {out.name: G.anchor.inputs[0].name}
    if G.kind == "matmul":
        a, b = G.anchor.inputs
        cd.meta.update(M=1 if len(a.shape) == 1 else a.shape[0], N=G.anchor.shape[-1], K=a.shape[-1],
                       wdtype=b.dtype, form=form)
    return cd
