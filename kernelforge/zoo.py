"""Graphs used by the tests, the benchmarks and the docs (small and benchmark-sized variants)."""
from __future__ import annotations

from .graph import Graph


def elementwise(m=48, n=96):
    g = Graph("elementwise")
    x = g.input("x", (m, n))
    b = g.input("b", (n,))
    y = g.gelu(x + b) * 0.5
    g.output(g.maximum(y, g.tanh(x)))
    return g


def rmsnorm(rows=8, n=896):
    g = Graph("rmsnorm")
    x = g.input("x", (rows, n))
    w = g.input("w", (n,))
    r = g.rsqrt(g.mean(x * x) + 1e-6)
    g.output(x * r * w)
    return g


def softmax(rows=16, n=1000):
    g = Graph("softmax")
    x = g.input("x", (rows, n))
    e = g.exp(x - g.max(x))
    g.output(e / g.sum(e))
    return g


def rowsum(rows=5, n=300):
    g = Graph("rowsum")
    x = g.input("x", (rows, n))
    g.output(g.sum(x))
    return g


def matvec(n=200, k=512, dtype="f32"):
    g = Graph(f"matvec_{dtype}")
    x = g.input("x", (k,))
    w = g.input("W", (n, k), dtype)
    g.output(g.linear(x, w))
    return g


def ffn_gate(n=256, k=512, dtype="q4_0"):
    """tinyinfer's SwiGLU: h = silu(W_gate x) * (W_up x); the silu*mul becomes a matvec epilogue."""
    g = Graph(f"ffn_{dtype}")
    x = g.input("x", (k,))
    wg = g.input("Wg", (n, k), dtype)
    wu = g.input("Wu", (n, k), dtype)
    up = g.linear(x, wu)
    g.output(g.silu(g.linear(x, wg)) * up)
    return g


def residual_matvec(n=192, k=256, dtype="q8_0"):
    g = Graph(f"residual_{dtype}")
    x = g.input("x", (k,))
    r = g.input("r", (n,))
    w = g.input("W", (n, k), dtype)
    g.output(g.linear(x, w) + r)
    return g


def gemm(m=70, n=96, k=64):
    g = Graph("gemm")
    a = g.input("A", (m, k))
    b = g.input("B", (k, n))
    bias = g.input("bias", (n,))
    g.output(g.relu(g.matmul(a, b) + bias))
    return g


def linear_gemm(m=64, n=128, k=96, dtype="f32"):
    g = Graph(f"linear_gemm_{dtype}")
    x = g.input("x", (m, k))
    w = g.input("W", (n, k), dtype)
    g.output(g.linear(x, w))
    return g


def jacobi(nx=14, ny=12, nz=10):
    """flipgpu pressure Jacobi step on the padded (nz+1, ny+1, nx+1) lattice."""
    g = Graph("jacobi")
    shape = (nz + 1, ny + 1, nx + 1)
    p = g.input("pressure", shape)
    rhs = g.input("rhs", shape)
    cell = g.input("cellType", shape, "u32")
    g.output(g.jacobi7(p, rhs, cell, (nz, ny, nx)))
    return g


def rbsor(nx=14, ny=12, nz=10, parity=0, omega=1.6):
    g = Graph(f"rbsor{parity}")
    shape = (nz + 1, ny + 1, nx + 1)
    p = g.input("pressure", shape)
    rhs = g.input("rhs", shape)
    cell = g.input("cellType", shape, "u32")
    g.output(g.rbsor(p, rhs, cell, (nz, ny, nx), parity, omega))
    return g


def chain_mlp(m=32, d=64, h=128):
    """Multi-kernel program: rmsnorm -> linear -> gelu -> linear (+ residual)."""
    g = Graph("mlp")
    x = g.input("x", (m, d))
    w = g.input("w", (d,))
    w1 = g.input("W1", (h, d))
    w2 = g.input("W2", (d, h))
    n = x * g.rsqrt(g.mean(x * x) + 1e-5) * w
    y = g.linear(g.gelu(g.linear(n, w1)), w2) + x
    g.output(y)
    return g


SMALL = {
    "elementwise": elementwise, "rmsnorm": lambda: rmsnorm(8, 96), "softmax": lambda: softmax(6, 100),
    "rowsum": rowsum, "matvec_f32": lambda: matvec(200, 512, "f32"),
    "matvec_f16": lambda: matvec(72, 256, "f16"), "matvec_q8_0": lambda: matvec(200, 512, "q8_0"),
    "matvec_q4_0": lambda: matvec(200, 512, "q4_0"), "ffn_q4_0": lambda: ffn_gate(96, 256, "q4_0"),
    "residual_q8_0": residual_matvec, "gemm": gemm, "linear_gemm_f32": lambda: linear_gemm(64, 128, 96, "f32"),
    "linear_gemm_q8_0": lambda: linear_gemm(48, 64, 96, "q8_0"), "jacobi": jacobi,
    "rbsor0": lambda: rbsor(parity=0), "rbsor1": lambda: rbsor(parity=1), "mlp": chain_mlp,
}
