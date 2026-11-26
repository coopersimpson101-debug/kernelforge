"""Extract flipgpu's hand-written pressure-solver kernels into a standalone WGSL module.

    python scripts/extract_flipgpu.py <path-to-flipgpu-checkout> [-o bench/handwritten/flipgpu_pressure.wgsl]

Copies, byte for byte, from flipgpu's src/gpu/shaders/flip.wgsl: the Params struct, the cell-type
constants, the bindings the pressure kernels use (uniform P, cellType, pressure, pressureOut, rhs),
the indexing helpers and the kernels neighbourSum / jacobi / rbUpdate / rbRed / rbBlack. Nothing is
rewritten; unused bindings (particles, velocity grids, hash grid) are dropped so the module compiles
on its own, and the binding numbers are kept.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

KEEP_BINDINGS = {0, 6, 7, 8, 9}
FUNCS = ["NX", "NY", "numNodes", "node", "nodeI", "coordOf", "inCells", "typeAt", "neighbourSum",
         "jacobi", "rbUpdate", "rbRed", "rbBlack"]


def block_at(src: str, start: int) -> int:
    """Index just past the brace block that starts at the first '{' after ``start``."""
    i = src.index("{", start)
    depth = 0
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def extract(src: str) -> str:
    out = []
    m = re.search(r"struct Params \{", src)
    out.append(src[m.start():block_at(src, m.start())] + ";")
    for c in ("FLUID", "AIR", "SOLID"):
        m = re.search(rf"const {c}: u32 = [^;]+;", src)
        out.append(m.group(0))
    for m in re.finditer(r"@group\(0\) @binding\((\d+)\) var[^;]+;", src):
        if int(m.group(1)) in KEEP_BINDINGS:
            out.append(m.group(0))
    for f in FUNCS:
        m = re.search(rf"(@compute @workgroup_size\(\d+\)\s*\n)?fn {f}\(", src)
        if m is None:
            raise SystemExit(f"function {f} not found")
        end = block_at(src, m.start())
        out.append(src[m.start():end])
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flipgpu")
    ap.add_argument("-o", default=str(Path(__file__).resolve().parent.parent / "bench/handwritten/flipgpu_pressure.wgsl"))
    a = ap.parse_args()
    src = (Path(a.flipgpu) / "src/gpu/shaders/flip.wgsl").read_text()
    head = ("// Extracted verbatim from flipgpu src/gpu/shaders/flip.wgsl by scripts/extract_flipgpu.py\n"
            "// (pressure solve only: Jacobi + red-black SOR). Hand-written baseline; do not edit.\n")
    Path(a.o).write_text(head + extract(src), newline="\n")
    print(f"wrote {a.o}")


if __name__ == "__main__":
    main()
