"""Apple-GPU twin of run_bench.py (needs macOS + build/kf_metal_run):

    clang -fobjc-arc -O2 -framework Metal -framework Foundation metal/kf_metal_run.m -o build/kf_metal_run
    python bench/run_bench_metal.py            # -> bench/results/m1.json

matvec: tinyinfer's ORIGINAL Metal kernels (bench/handwritten/tinyinfer_kernels.metal) vs generated
MSL, default schedule and autotuned ON THE MAC (same tuner, Metal harness). Pressure solve:
flipgpu is WebGPU-only, so only generated default vs tuned are reported there.
Not run by the author yet: the README's M1 column stays empty until this is run on an M1.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import zoo  # noqa: E402
from kernelforge.compiler import compile_group  # noqa: E402
from kernelforge.graph import evaluate, random_feeds  # noqa: E402
from kernelforge.runtime import GpuJob  # noqa: E402
from kernelforge.target import M1_MSL  # noqa: E402
from kernelforge.tune import feeds_to_bytes, single_group, tune_group  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from run_bench import MATVEC_SHAPES, rel_err, repeat_for, wbytes  # noqa: E402

RUNS = 21
HW = (ROOT / "bench/handwritten/tinyinfer_kernels.metal").read_text()


def main():
    rows = []
    adapter = None
    for label, n, k in MATVEC_SHAPES:
        for dtype in ("f32", "q8_0", "q4_0"):
            g = zoo.matvec(n, k, dtype)
            f = random_feeds(g, 11)
            out = g.outputs[0]
            ref = evaluate(g, f)[out.name]
            G = single_group(g)
            fb = feeds_to_bytes(g, f)
            r = tune_group(G, fb, ref, M1_MSL, backend="msl")
            R = repeat_for(r.best_s)
            job = GpuJob("msl")
            for name, b in fb.items():
                job.buffer(name, b)
            job.buffer(out.name, 4 * n)
            job.buffer("y_hw", 4 * n)
            job.buffer("args", struct.pack("<ii", n, k) + bytes(8))
            job.case("hand", [{"name": "h", "code": HW, "entry": f"matvec_{dtype}"}],
                     [{"pipeline": "h", "bindings": ["W", "x", "y_hw", "args"], "groups": [-(-n // 8), 1, 1],
                       "threads": [256, 1, 1]}], readback=["y_hw"], repeat=R, runs=RUNS)
            for cid, kn in (("gen_default", None), ("gen_tuned", r.best)):
                ck = compile_group(G, M1_MSL, kn)
                job.case(cid, [{"name": "p", "code": ck.source, "entry": ck.kernel.name}],
                         [{"pipeline": "p", "bindings": ck.params, "groups": list(ck.grid),
                           "threads": list(ck.workgroup)}], readback=[ck.output], repeat=R, runs=RUNS)
            res = job.run()
            adapter = res["adapter"]
            entry = {"family": "matvec", "name": label, "shape": [n, k], "dtype": dtype, "metric": "GB/s",
                     "knobs": r.best}
            for c in res["cases"]:
                if not c["ok"]:
                    raise RuntimeError(c.get("error"))
                nm = "y_hw" if c["id"] == "hand" else out.name
                got = np.frombuffer(c["data"][nm], np.float32)[:n]
                s = c["median_ns"] * 1e-9
                entry[c["id"]] = {"us": s * 1e6, "value": (wbytes(dtype, n, k) + 4 * (n + k)) / s / 1e9,
                                  "rel_err": rel_err(got, ref), "timer": c["timer"]}
            print(entry, flush=True)
            rows.append(entry)
            job.cleanup()
    meta = {"date": dt.date.today().isoformat(), "adapter": adapter, "host": platform.platform(), "runs": RUNS}
    p = ROOT / "bench/results/m1.json"
    p.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
