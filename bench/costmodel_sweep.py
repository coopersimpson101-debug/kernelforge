"""How good is the cost model? Time EVERY candidate of a few knob spaces and compare.

    python bench/costmodel_sweep.py      # -> bench/results/costmodel_rtx2060.json

Reports Spearman rank correlation of predicted vs measured time, the model's rank of the truly
fastest schedule, and how much slower the tuner's pick (best of the model's top 16) is than the
true optimum ("regret"; 1.00x = found it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import zoo  # noqa: E402
from kernelforge.graph import evaluate, random_feeds  # noqa: E402
from kernelforge.target import RTX2060_WGSL  # noqa: E402
from kernelforge.tune import feeds_to_bytes, single_group, spearman, tune_group  # noqa: E402

CASES = [("matvec q4_0 4864x896", lambda: zoo.matvec(4864, 896, "q4_0")),
         ("matvec f32 4864x896", lambda: zoo.matvec(4864, 896, "f32")),
         ("jacobi 64^3", lambda: zoo.jacobi(64, 64, 64))]


def main():
    out = []
    for name, mk in CASES:
        g = mk()
        f = random_feeds(g, 21)
        ref = evaluate(g, f)[g.outputs[0].name]
        r = tune_group(single_group(g), feeds_to_bytes(g, f), ref, RTX2060_WGSL, full=True, runs=5)
        rows = [x for x in r.log[1:] if x["ok"] and x["meas_s"]]
        pred = [x["pred_s"] for x in rows]
        meas = [x["meas_s"] for x in rows]
        order = sorted(range(len(rows)), key=lambda i: pred[i])
        true_best = min(range(len(rows)), key=lambda i: meas[i])
        top = min(meas[i] for i in order[:16])
        out.append({"kernel": name, "n": len(rows), "spearman": spearman(pred, meas),
                    "best_rank_by_model": order.index(true_best) + 1, "topk_regret": top / meas[true_best],
                    "best_knobs": rows[true_best]["knobs"], "best_us": meas[true_best] * 1e6,
                    "rejected": sum(1 for x in r.log if not x["ok"]),
                    "points": [[p, m] for p, m in zip(pred, meas)]})
        print({k: v for k, v in out[-1].items() if k != "points"}, flush=True)
    p = ROOT / "bench/results/costmodel_rtx2060.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
