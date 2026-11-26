"""Where does the quantised-matvec speed-up come from? Sweep lanes-per-row T (rows per workgroup
R = 256 / T, everything else fixed) for the generated kernel, next to tinyinfer's kernel (T = 32).

    python bench/lanes_ablation.py      # -> bench/results/lanes_rtx2060.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import baselines, zoo  # noqa: E402
from kernelforge.compiler import compile_group  # noqa: E402
from kernelforge.graph import evaluate, random_feeds  # noqa: E402
from kernelforge.runtime import GpuJob  # noqa: E402
from kernelforge.target import RTX2060_WGSL  # noqa: E402
from kernelforge.tune import feeds_to_bytes, single_group  # noqa: E402

N, K = 4864, 896
LANES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def main():
    out = []
    for dtype in ("q4_0", "q8_0", "f32"):
        g = zoo.matvec(N, K, dtype)
        f = random_feeds(g, 3)
        G = single_group(g)
        fb = feeds_to_bytes(g, f)
        job = GpuJob()
        for n, b in fb.items():
            job.buffer(n, b)
        job.buffer(G.output.name, 4 * N)
        job.buffer("y_hw", 4 * N)
        pipes, steps, extra = baselines.tinyinfer_matvec_case(dtype, N, K, "W", "x", "y_hw")
        for n, b in extra.items():
            job.buffer(n, b)
        job.case("hand", pipes, steps, readback=["y_hw"], repeat=4, runs=31)
        for T in LANES:
            kn = {"R": 256 // T, "T": T, "U": 2, "V": 1, "xlane": "shared", "stage_x": 0, "unroll": 1}
            ck = compile_group(G, RTX2060_WGSL, kn)
            job.case(f"T{T}", [{"name": "p", "code": ck.source, "entry": ck.kernel.name}],
                     [{"pipeline": "p", "bindings": ck.params, "groups": list(ck.grid)}],
                     readback=[ck.output], repeat=4, runs=31)
        res = job.run()
        job.cleanup()
        row = {"dtype": dtype, "shape": [N, K]}
        for c in res["cases"]:
            assert c["ok"], c.get("error")
            row[c["id"]] = c["median_ns"] / 1e3
        print(row, flush=True)
        out.append(row)
    p = ROOT / "bench/results/lanes_rtx2060.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
