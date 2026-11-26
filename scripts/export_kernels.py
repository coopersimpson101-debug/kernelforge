"""Write the benchmark kernels as source files: WGSL (schedules autotuned on the RTX 2060),
MSL (schedule chosen by the cost model for the M1 target: NOT tuned on hardware) and C (oracle).

    python scripts/export_kernels.py -o build/kernels
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import zoo  # noqa: E402
from kernelforge.compiler import compile_group, form_for  # noqa: E402
from kernelforge.compute import compute_def  # noqa: E402
from kernelforge.cost import predict  # noqa: E402
from kernelforge.fusion import fuse  # noqa: E402
from kernelforge.schedule import ScheduleError, space, template_for  # noqa: E402
from kernelforge.target import C_REF, M1_MSL, RTX2060_WGSL  # noqa: E402


def graph_for(key: str):
    m = re.match(r"(matvec|ffn-up|ffn-gate\+silu\*mul|gemm) (\w+) ([\dx]+)$", key)
    if m:
        kind, dt, dims = m.group(1), m.group(2), [int(v) for v in m.group(3).split("x")]
        if kind == "matvec":
            return zoo.matvec(dims[0], dims[1], dt), 0
        if kind == "gemm":
            return zoo.linear_gemm(dims[0], dims[1], dims[2], dt), 0
        return zoo.ffn_gate(dims[0], dims[1], dt), 0 if kind == "ffn-up" else 1
    m = re.match(r"(jacobi|rbsor-red) (\d+)\^3$", key)
    if m:
        n = int(m.group(2))
        return (zoo.jacobi(n, n, n) if m.group(1) == "jacobi" else zoo.rbsor(n, n, n, 0)), 0
    raise ValueError(key)


def cost_best(G, t):
    cd = compute_def(G, form_for(G))
    tmpl = template_for(cd)
    for _, kn in sorted(((predict(tmpl, cd, kn, t), kn) for kn in space(tmpl, cd, t)), key=lambda x: x[0]):
        try:
            compile_group(G, t, kn)
            return kn
        except ScheduleError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=str(ROOT / "build/kernels"))
    ap.add_argument("--tuning", default=str(ROOT / "bench/results/rtx2060_tuning.json"))
    a = ap.parse_args()
    out = Path(a.o)
    tun = json.loads(Path(a.tuning).read_text()) if Path(a.tuning).exists() else {}
    keys = list(tun) or ["matvec q4_0 4864x896", "jacobi 64^3"]
    for sub in ("wgsl", "msl", "c"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    index = []
    for key in keys:
        g, gi = graph_for(key)
        G = fuse(g)[gi]
        best = tun.get(key, {}).get("best")
        slug = re.sub(r"[^\w]+", "_", key).strip("_")
        w = compile_group(G, RTX2060_WGSL, best)
        mk = cost_best(G, M1_MSL)
        m = compile_group(G, M1_MSL, mk)
        c = compile_group(G, C_REF, None)
        (out / "wgsl" / f"{slug}.wgsl").write_text(w.source)
        (out / "msl" / f"{slug}.metal").write_text(m.source)
        (out / "c" / f"{slug}.c").write_text(c.source)
        index.append({"kernel": key, "wgsl_knobs (tuned on RTX 2060)": best, "msl_knobs (cost model, M1)": mk})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} kernels x 3 backends to {out}")


if __name__ == "__main__":
    main()
