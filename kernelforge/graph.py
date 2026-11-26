"""Tensor graph IR: a DAG of ops with static shapes and dtypes, plus a NumPy reference interpreter.

The interpreter is deliberately independent of the compiler (it never looks at compute definitions),
so it is a real oracle: every generated kernel is compared against it on random inputs.

Storage dtypes: f32, f16, u32 (cell flags), q8_0, q4_0 (tinyinfer/ggml blocks, last axis % 32 == 0).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import quant

UNARY = ("neg", "exp", "silu", "relu", "sqrt", "rsqrt", "tanh", "sigmoid", "gelu", "abs")
BINARY = ("add", "sub", "mul", "div", "maximum", "minimum")
SCALAR = ("adds", "muls")
REDUCE = ("sum", "max", "mean")
ANCHOR = ("matmul", "linear")
STENCIL = ("jacobi7", "rbsor")
FLUID, AIR, SOLID = 0, 1, 2

_uid = itertools.count()


@dataclass(eq=False)
class Node:
    op: str
    inputs: list
    shape: tuple
    dtype: str = "f32"
    attrs: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.op}{next(_uid)}"

    def __repr__(self):
        return f"%{self.name}:{self.op}{list(self.shape)}"

    # sugar so graphs read like numpy
    def __add__(self, o): return self.g._bin("add", self, o)
    def __sub__(self, o): return self.g._bin("sub", self, o)
    def __mul__(self, o): return self.g._bin("mul", self, o)
    def __truediv__(self, o): return self.g._bin("div", self, o)


def broadcast_shape(a: tuple, b: tuple) -> tuple:
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            raise ValueError(f"cannot broadcast {a} and {b}")
        out.append(max(x, y))
    return tuple(out)


class Graph:
    def __init__(self, name: str = "g"):
        self.name = name
        self.nodes: list[Node] = []
        self.outputs: list[Node] = []

    def _add(self, n: Node) -> Node:
        n.g = self
        self.nodes.append(n)
        return n

    # ---------------------------------------------------------------- builders
    def input(self, name: str, shape, dtype: str = "f32") -> Node:
        shape = tuple(int(s) for s in shape)
        if dtype in quant.BLOCK_BYTES:
            assert shape[-1] % quant.QK == 0, "quantised tensors need last dim % 32 == 0"
        return self._add(Node("input", [], shape, dtype, name=name))

    def _bin(self, op, a, b) -> Node:
        if not isinstance(b, Node):
            return self._add(Node("adds" if op in ("add", "sub") else "muls", [a], a.shape,
                                  attrs={"c": float(b) if op in ("add", "mul") else
                                         (-float(b) if op == "sub" else 1.0 / float(b))}))
        return self._add(Node(op, [a, b], broadcast_shape(a.shape, b.shape)))

    def add(self, a, b): return self._bin("add", a, b)
    def sub(self, a, b): return self._bin("sub", a, b)
    def mul(self, a, b): return self._bin("mul", a, b)
    def div(self, a, b): return self._bin("div", a, b)
    def maximum(self, a, b): return self._add(Node("maximum", [a, b], broadcast_shape(a.shape, b.shape)))
    def minimum(self, a, b): return self._add(Node("minimum", [a, b], broadcast_shape(a.shape, b.shape)))

    def unary(self, op: str, a: Node) -> Node:
        assert op in UNARY
        return self._add(Node(op, [a], a.shape))

    def __getattr__(self, op):
        if op in UNARY:
            return lambda a: self.unary(op, a)
        raise AttributeError(op)

    def reduce(self, op: str, a: Node) -> Node:
        assert op in REDUCE
        return self._add(Node(op, [a], a.shape[:-1] + (1,)))

    def sum(self, a): return self.reduce("sum", a)
    def max(self, a): return self.reduce("max", a)
    def mean(self, a): return self.reduce("mean", a)

    def matmul(self, a: Node, b: Node) -> Node:
        (m, k), (k2, n) = a.shape, b.shape
        assert k == k2
        return self._add(Node("matmul", [a, b], (m, n)))

    def linear(self, x: Node, w: Node) -> Node:
        """y[..., n] = sum_k x[..., k] * w[n, k]  (tinyinfer/ggml weight layout)."""
        assert x.shape[-1] == w.shape[1]
        return self._add(Node("linear", [x, w], x.shape[:-1] + (w.shape[0],)))

    def jacobi7(self, p: Node, rhs: Node, cell: Node, dims) -> Node:
        """flipgpu's Jacobi pressure step on the padded (nz+1, ny+1, nx+1) lattice."""
        return self._add(Node("jacobi7", [p, rhs, cell], p.shape, attrs={"dims": tuple(dims)}))

    def rbsor(self, p: Node, rhs: Node, cell: Node, dims, parity: int, omega: float) -> Node:
        """One colour of flipgpu's red-black SOR sweep (in place on p)."""
        return self._add(Node("rbsor", [p, rhs, cell], p.shape,
                              attrs={"dims": tuple(dims), "parity": parity, "omega": float(omega)}))

    def output(self, *ns: Node):
        self.outputs.extend(ns)
        return ns[0] if len(ns) == 1 else ns

    # ---------------------------------------------------------------- queries
    def inputs(self) -> list[Node]:
        return [n for n in self.nodes if n.op == "input"]

    def consumers(self) -> dict:
        c: dict = {n: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                c[i].append(n)
        return c

    def __repr__(self):
        lines = [f"graph {self.name}:"]
        for n in self.nodes:
            ins = ", ".join("%" + i.name for i in n.inputs)
            at = "".join(f" {k}={v}" for k, v in n.attrs.items())
            lines.append(f"  %{n.name}: {n.dtype}{list(n.shape)} = {n.op}({ins}){at}")
        lines.append("  return " + ", ".join("%" + o.name for o in self.outputs))
        return "\n".join(lines)


# ----------------------------------------------------------------------------- reference interpreter

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


_UN = {
    "neg": lambda x: -x, "exp": np.exp, "silu": lambda x: x / (1.0 + np.exp(-x)),
    "relu": lambda x: np.maximum(x, 0.0), "sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "tanh": np.tanh, "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)), "gelu": _gelu, "abs": np.abs,
}
_BIN = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
        "maximum": np.maximum, "minimum": np.minimum}


def stencil_ref(op: str, p, rhs, cell, dims, parity=0, omega=1.0):
    """NumPy port of flipgpu's jacobi / rbUpdate on the padded lattice (float32, same op order)."""
    p = p.astype(np.float32)
    rhs = rhs.astype(np.float32)
    nz, ny, nx = dims
    Z, Y, X = p.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")

    def in_cells(z, y, x):
        return (z >= 0) & (y >= 0) & (x >= 0) & (z < nz) & (y < ny) & (x < nx)

    def type_at(z, y, x):
        ok = in_cells(z, y, x)
        t = np.full(p.shape, SOLID, np.uint32)
        t[ok] = cell[z[ok], y[ok], x[ok]]
        return t

    def p_at(z, y, x):
        ok = in_cells(z, y, x)
        out = np.zeros(p.shape, np.float32)
        out[ok] = p[z[ok], y[ok], x[ok]]
        return out

    diag = np.zeros(p.shape, np.float32)
    s = np.zeros(p.shape, np.float32)
    for (dz, dy, dx) in ((0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)):
        t = type_at(zz + dz, yy + dy, xx + dx)
        diag = diag + (t != SOLID).astype(np.float32)
        s = s + np.where(t == FLUID, p_at(zz + dz, yy + dy, xx + dx), np.float32(0))
    fluid = type_at(zz, yy, xx) == FLUID
    with np.errstate(divide="ignore", invalid="ignore"):
        gs = np.where(diag > 0, (rhs + s) / np.where(diag > 0, diag, 1), np.float32(0)).astype(np.float32)
    if op == "jacobi7":
        return np.where(fluid, gs, np.float32(0)).astype(np.float32)
    par = ((xx + yy + zz) & 1) == parity
    upd = fluid & par & (diag > 0)
    w = np.float32(omega)
    return np.where(upd, (np.float32(1) - w) * p + w * gs, p).astype(np.float32)


def evaluate(g: Graph, feeds: dict) -> dict:
    """Run the graph in NumPy (float64 arithmetic for float ops). Returns {node_name: array}."""
    vals: dict = {}
    for n in g.nodes:
        if n.op == "input":
            v = feeds[n.name]
            if n.dtype in quant.BLOCK_BYTES:
                v = quant.dequantize(v, n.dtype, n.shape).astype(np.float64)
            elif n.dtype == "f16":
                v = np.asarray(v, np.float16).astype(np.float64)
            elif n.dtype == "u32":
                v = np.asarray(v, np.uint32)
            else:
                v = np.asarray(v, np.float32).astype(np.float64)
            vals[n] = v.reshape(n.shape)
            continue
        a = [vals[i] for i in n.inputs]
        if n.op in _UN:
            r = _UN[n.op](a[0])
        elif n.op in _BIN:
            r = _BIN[n.op](a[0], a[1])
        elif n.op == "adds":
            r = a[0] + n.attrs["c"]
        elif n.op == "muls":
            r = a[0] * n.attrs["c"]
        elif n.op == "sum":
            r = a[0].sum(axis=-1, keepdims=True)
        elif n.op == "max":
            r = a[0].max(axis=-1, keepdims=True)
        elif n.op == "mean":
            r = a[0].mean(axis=-1, keepdims=True)
        elif n.op == "matmul":
            r = a[0] @ a[1]
        elif n.op == "linear":
            r = a[0] @ a[1].T
        elif n.op in STENCIL:
            r = stencil_ref(n.op, a[0], a[1], a[2], n.attrs["dims"], n.attrs.get("parity", 0),
                            n.attrs.get("omega", 1.0)).astype(np.float64)
        else:
            raise NotImplementedError(n.op)
        vals[n] = np.asarray(r).reshape(n.shape)
    return {n.name: v for n, v in vals.items()}


def random_feeds(g: Graph, seed: int = 0) -> dict:
    """Random inputs: N(0,1) floats, pre-quantised blocks for q8_0/q4_0, plausible cell flags."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for n in g.inputs():
        if n.dtype in quant.BLOCK_BYTES:
            w = rng.standard_normal(n.shape).astype(np.float32)
            feeds[n.name] = (quant.quantize_q8_0 if n.dtype == "q8_0" else quant.quantize_q4_0)(w)
        elif n.dtype == "u32":
            Z = n.shape[0]
            t = np.where(rng.random(n.shape) < 0.85, FLUID, AIR).astype(np.uint32)
            t[: Z // 6] = np.where(rng.random(t[: Z // 6].shape) < 0.5, SOLID, t[: Z // 6])
            feeds[n.name] = t
        elif n.dtype == "f16":
            feeds[n.name] = rng.standard_normal(n.shape).astype(np.float16)
        else:
            feeds[n.name] = rng.standard_normal(n.shape).astype(np.float32)
    return feeds
