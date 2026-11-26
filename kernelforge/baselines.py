"""Hand-written baseline kernels (bench/handwritten/*.wgsl) wired into harness cases.

* tinyinfer: WGSL ports of its Metal matvec_f32 / matvec_q8_0 / matvec_q4_0 / silu_mul;
* flipgpu:   its own WGSL jacobi / rbRed / rbBlack, extracted verbatim by scripts/extract_flipgpu.py.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parent.parent / "bench" / "handwritten"


def source(name: str) -> str:
    return (HW / name).read_text()


def tinyinfer_matvec_case(dtype: str, rows: int, cols: int, w: str = "W", x: str = "x", y: str = "y",
                          args: str = "args"):
    """(pipelines, steps, extra_buffers) for y = W x with tinyinfer's kernel (8 rows / threadgroup)."""
    entry = f"matvec_{dtype}"
    pipes = [{"name": entry, "code": source(f"tinyinfer_matvec_{dtype}.wgsl"), "entry": entry}]
    steps = [{"pipeline": entry, "bindings": [w, x, y, args], "groups": [-(-rows // 8), 1, 1]}]
    return pipes, steps, {args: struct.pack("<ii", rows, cols) + bytes(8)}


def tinyinfer_silu_mul_case(n: int, g: str, u: str, nbuf: str = "n_silu"):
    pipes = [{"name": "silu_mul", "code": source("tinyinfer_silu_mul.wgsl"), "entry": "silu_mul"}]
    steps = [{"pipeline": "silu_mul", "bindings": [g, u, nbuf], "groups": [-(-n // 256), 1, 1]}]
    return pipes, steps, {nbuf: struct.pack("<i", n) + bytes(12)}


def flipgpu_params(dims, omega: float) -> bytes:
    """flipgpu's Params uniform (7 x vec4): dims=(nx, ny, nz, 0), p1.y = SOR omega."""
    nz, ny, nx = dims
    b = bytearray(7 * 16)
    struct.pack_into("<4I", b, 0, nx, ny, nz, 0)
    struct.pack_into("<4f", b, 5 * 16, 0.0, omega, 0.0, 0.0)
    return bytes(b)


def flipgpu_case(kind: str, dims, p="pressure", pout="pressureOut", rhs="rhs", cell="cellType",
                 params="P"):
    """kind: 'jacobi' (one step pressure -> pressureOut) or 'rb' (rbRed + rbBlack, in place)."""
    nz, ny, nx = dims
    nodes = (nx + 1) * (ny + 1) * (nz + 1)
    code = source("flipgpu_pressure.wgsl")
    groups = [-(-nodes // 64), 1, 1]
    if kind == "jacobi":
        pipes = [{"name": "jacobi", "code": code, "entry": "jacobi"}]
        steps = [{"pipeline": "jacobi", "bindings": {"0": params, "6": cell, "7": p, "8": pout, "9": rhs},
                  "groups": groups}]
    else:
        pipes = [{"name": "rbRed", "code": code, "entry": "rbRed"}, {"name": "rbBlack", "code": code, "entry": "rbBlack"}]
        b = {"0": params, "6": cell, "7": p, "9": rhs}
        steps = [{"pipeline": "rbRed", "bindings": b, "groups": groups},
                 {"pipeline": "rbBlack", "bindings": b, "groups": groups}]
    return pipes, steps


def f32(a) -> bytes:
    return np.ascontiguousarray(a, np.float32).tobytes()
