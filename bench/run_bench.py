"""Generated vs hand-written kernels, measured on this machine's GPU through Dawn/WebGPU.

    python bench/run_bench.py                 # full suite -> bench/results/rtx2060.json
    python bench/run_bench.py --quick         # 1 shape per family (smoke test, ~2 min)
    python bench/run_bench.py --only matvec   # one family: matvec | ffn | gemm | pressure

For every case the hand-written kernel, the generated kernel with the default (untuned) schedule
and the generated kernel with the autotuned schedule run in ONE harness process on the SAME
buffers; each is timed as the median of N runs, every run being one compute pass of R back-to-back
repetitions bracketed by GPU timestamp queries. Every output is checked against NumPy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import baselines, zoo  # noqa: E402
from kernelforge.compiler import compile_group  # noqa: E402
from kernelforge.graph import evaluate, random_feeds, stencil_ref  # noqa: E402
from kernelforge.passes import storage_size  # noqa: E402
from kernelforge.quant import BLOCK_BYTES, QK  # noqa: E402
from kernelforge.runtime import GpuJob, to_bytes  # noqa: E402
from kernelforge.schedule import default_knobs  # noqa: E402
from kernelforge.target import RTX2060_WGSL  # noqa: E402
from kernelforge.tune import feeds_to_bytes, single_group, tune_group  # noqa: E402

RUNS = 41
T = RTX2060_WGSL
TUNE_LOG: dict = {}

# tinyinfer's shapes (Qwen2.5-0.5B and -1.5B decode): rows x cols of y = W x
MATVEC_SHAPES = [("0.5B attn q/o", 896, 896), ("0.5B ffn gate/up", 4864, 896), ("0.5B ffn down", 896, 4864),
                 ("1.5B ffn gate/up", 8960, 1536), ("1.5B ffn down", 1536, 8960)]


def wbytes(dtype, rows, cols):
    if dtype in BLOCK_BYTES:
        return rows * cols // QK * BLOCK_BYTES[dtype]
    return rows * cols * (2 if dtype == "f16" else 4)


def rel_err(got, ref):
    return float(np.abs(np.asarray(got, np.float64) - ref).max() / max(np.abs(ref).max(), 1e-12))


def repeat_for(seconds: float) -> int:
    """Repetitions per timed pass: ~0.3 ms of GPU work (short passes are less exposed to
    time-slicing with other processes on a shared GPU)."""
    return int(max(2, min(50, 3e-4 / max(seconds, 1e-7))))


def tuned(G, fb, ref, key, top_k=16, full=False):
    t0 = time.time()
    r = tune_group(G, fb, ref, T, top_k=top_k, full=full, log=lambda m: print(m, flush=True))
    TUNE_LOG[key] = {"template": r.template, "best": r.best, "best_s": r.best_s, "default_s": r.default_s,
                     "space": r.space_size, "measured": len(r.log), "seconds": round(time.time() - t0, 1),
                     "log": r.log}
    print(f"  tuned {key}: {r.best} ({r.best_s * 1e6:.1f} us; default {r.default_s * 1e6:.1f} us; "
          f"{len(r.log)}/{r.space_size} measured)", flush=True)
    return r.best


def gen_case(job, cid, ck, repeat):
    job.case(cid, [{"name": "p", "code": ck.source, "entry": ck.kernel.name}],
             [{"pipeline": "p", "bindings": ck.params, "groups": list(ck.grid)}],
             readback=[ck.output], repeat=repeat, runs=RUNS)


UTIL: list = []


def gpu_util():
    """GPU utilisation (%) reported by nvidia-smi right before our job starts, i.e. load from
    OTHER processes (None if nvidia-smi is not available)."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def finish(job):
    UTIL.append(gpu_util())
    try:
        res = job.run(timeout=3600)
        if res["errors"]:
            raise RuntimeError(res["errors"])
        out = {}
        for c in res["cases"]:
            if not c["ok"]:
                raise RuntimeError(f"{c['id']}: {c.get('error')}")
            out[c["id"]] = c
        return out, res["adapter"]
    finally:
        job.cleanup()


def row(family, name, shape, dtype, metric, work, cases, errs, knobs, extra=None):
    """work: bytes (GB/s) or flops (GFLOP/s) or nodes (Gnode/s) per repetition."""
    r = {"family": family, "name": name, "shape": shape, "dtype": dtype, "metric": metric, "knobs": knobs}
    for k, c in cases.items():
        s = c["median_ns"] * 1e-9
        r[k] = {"us": s * 1e6, "value": work / s / 1e9, "min_us": min(c["times_ns"]) / 1e3, "rel_err": errs[k], "repeat": c["repeat"],
                "timer": c["timer"], "runs": len(c["times_ns"]),
                "iqr_us": [float(np.percentile(c["times_ns"], 25)) / 1e3, float(np.percentile(c["times_ns"], 75)) / 1e3]}
    if extra:
        r.update(extra)
    print(f"  {family:9s} {name:18s} {dtype:5s} " + "  ".join(
        f"{k}={r[k]['value']:.1f} {metric} ({r[k]['us']:.1f}us)" for k in cases), flush=True)
    return r


# ----------------------------------------------------------------------------- families

def bench_matvec(shapes, dtypes):
    rows_out = []
    for (label, n, k) in shapes:
        for dtype in dtypes:
            g = zoo.matvec(n, k, dtype)
            f = random_feeds(g, 11)
            out = g.outputs[0]
            ref = evaluate(g, f)[out.name]
            G = single_group(g)
            fb = feeds_to_bytes(g, f)
            best = tuned(G, fb, ref, f"matvec {dtype} {n}x{k}", top_k=32)
            ck_def = compile_group(G, T, None)
            ck_tun = compile_group(G, T, best)
            est = TUNE_LOG[f"matvec {dtype} {n}x{k}"]["best_s"] or 1e-4
            R = repeat_for(est)
            job = GpuJob()
            for name, b in fb.items():
                job.buffer(name, b)
            job.buffer(out.name, 4 * n)
            pipes, steps, extra = baselines.tinyinfer_matvec_case(dtype, n, k, "W", "x", "y_hw")
            for name, b in extra.items():
                job.buffer(name, b)
            job.buffer("y_hw", 4 * n)
            job.case("hand", pipes, steps, readback=["y_hw"], repeat=R, runs=RUNS)
            gen_case(job, "gen_default", ck_def, R)
            gen_case(job, "gen_tuned", ck_tun, R)
            res, adapter = finish(job)
            get = lambda c, nm: np.frombuffer(res[c]["data"][nm], np.float32)[:n]
            errs = {"hand": rel_err(get("hand", "y_hw"), ref), "gen_default": rel_err(get("gen_default", out.name), ref),
                    "gen_tuned": rel_err(get("gen_tuned", out.name), ref)}
            rows_out.append(row("matvec", label, [n, k], dtype, "GB/s", wbytes(dtype, n, k) + 4 * (n + k), res, errs,
                                best, {"gflops_tuned": 2 * n * k / (res["gen_tuned"]["median_ns"] * 1e-9) / 1e9}))
    return rows_out, adapter


def bench_ffn(shapes, dtype="q4_0"):
    """SwiGLU block: hand = matvec(gate) + matvec(up) + silu_mul (tinyinfer's 3 dispatches);
    generated = matvec(up) + matvec(gate) with silu(.)*up fused as its epilogue (2 dispatches)."""
    rows_out, adapter = [], None
    for (label, n, k) in shapes:
        g = zoo.ffn_gate(n, k, dtype)
        f = random_feeds(g, 12)
        vals = evaluate(g, f)
        out = g.outputs[0]
        ref = vals[out.name]
        from kernelforge.fusion import fuse
        groups = fuse(g)
        assert len(groups) == 2, groups
        up_G, gate_G = groups
        fb = feeds_to_bytes(g, f)
        fb_gate = dict(fb)
        fb_gate[up_G.output.name] = np.asarray(vals[up_G.output.name], np.float32).tobytes()
        kb_up = tuned(up_G, fb, vals[up_G.output.name], f"ffn-up {dtype} {n}x{k}")
        kb_gate = tuned(gate_G, fb_gate, ref, f"ffn-gate+silu*mul {dtype} {n}x{k}")
        cks = {"gen_default": [compile_group(up_G, T, None), compile_group(gate_G, T, None)],
               "gen_tuned": [compile_group(up_G, T, kb_up), compile_group(gate_G, T, kb_gate)]}
        R = repeat_for(TUNE_LOG[f"ffn-up {dtype} {n}x{k}"]["best_s"] * 2)
        job = GpuJob()
        for name, b in fb.items():
            job.buffer(name, b)
        for name in (up_G.output.name, out.name, "g_hw", "u_hw"):
            job.buffer(name, 4 * n)
        p1, s1, e1 = baselines.tinyinfer_matvec_case(dtype, n, k, "Wg", "x", "g_hw", "args")
        p2, s2, _ = baselines.tinyinfer_matvec_case(dtype, n, k, "Wu", "x", "u_hw", "args")
        p3, s3, e3 = baselines.tinyinfer_silu_mul_case(n, "g_hw", "u_hw")
        for name, b in {**e1, **e3}.items():
            job.buffer(name, b)
        p2 = [dict(p, name="mv2") for p in p2]
        s2 = [dict(s, pipeline="mv2") for s in s2]
        job.case("hand", p1 + p2 + p3, s1 + s2 + s3, readback=["g_hw"], repeat=R, runs=RUNS)
        for cid, (a, b) in cks.items():
            job.case(cid, [{"name": "a", "code": a.source, "entry": a.kernel.name},
                           {"name": "b", "code": b.source, "entry": b.kernel.name}],
                     [{"pipeline": "a", "bindings": a.params, "groups": list(a.grid)},
                      {"pipeline": "b", "bindings": b.params, "groups": list(b.grid)}],
                     readback=[out.name], repeat=R, runs=RUNS)
        res, adapter = finish(job)
        get = lambda c, nm: np.frombuffer(res[c]["data"][nm], np.float32)[:n]
        errs = {"hand": rel_err(get("hand", "g_hw"), ref), "gen_default": rel_err(get("gen_default", out.name), ref),
                "gen_tuned": rel_err(get("gen_tuned", out.name), ref)}
        rows_out.append(row("ffn", label, [n, k], dtype, "GB/s", 2 * wbytes(dtype, n, k) + 4 * (k + n), res, errs,
                            {"up": kb_up, "gate": kb_gate}, {"dispatches": {"hand": 3, "gen": 2}}))
    return rows_out, adapter


def bench_gemm(cases):
    rows_out, adapter = [], None
    for (label, m, n, k, dtype) in cases:
        g = zoo.linear_gemm(m, n, k, dtype)
        f = random_feeds(g, 13)
        out = g.outputs[0]
        ref = evaluate(g, f)[out.name]
        G = single_group(g)
        fb = feeds_to_bytes(g, f)
        best = tuned(G, fb, ref, f"gemm {dtype} {m}x{n}x{k}", top_k=24)
        R = repeat_for(TUNE_LOG[f"gemm {dtype} {m}x{n}x{k}"]["best_s"])
        job = GpuJob()
        for name, b in fb.items():
            job.buffer(name, b)
        job.buffer(out.name, 4 * m * n)
        gen_case(job, "gen_default", compile_group(G, T, None), max(2, R // 8))
        gen_case(job, "gen_tuned", compile_group(G, T, best), R)
        res, adapter = finish(job)
        errs = {c: rel_err(np.frombuffer(res[c]["data"][out.name], np.float32)[: m * n].reshape(m, n), ref)
                for c in ("gen_default", "gen_tuned")}
        rows_out.append(row("gemm", label, [m, n, k], dtype, "GFLOP/s", 2.0 * m * n * k, res, errs, best))
    return rows_out, adapter


def bench_pressure(sizes):
    rows_out, adapter = [], None
    for c3 in sizes:
        dims = (c3, c3, c3)
        nodes = (c3 + 1) ** 3
        for kind in ("jacobi", "rb"):
            if kind == "jacobi":
                g = zoo.jacobi(c3, c3, c3)
                gs = [g]
            else:
                gs = [zoo.rbsor(c3, c3, c3, 0, 1.6), zoo.rbsor(c3, c3, c3, 1, 1.6)]
                g = gs[0]
            f = random_feeds(g, 14)
            if kind == "jacobi":
                ref = stencil_ref("jacobi7", f["pressure"], f["rhs"], f["cellType"], dims)
            else:
                red = stencil_ref("rbsor", f["pressure"], f["rhs"], f["cellType"], dims, 0, 1.6)
                ref = stencil_ref("rbsor", red, f["rhs"], f["cellType"], dims, 1, 1.6)
            fb = feeds_to_bytes(g, f)
            Gs = [single_group(x) for x in gs]
            if kind == "jacobi":
                # exhaustive: the stencil space is small (136) and the cost model ranks it poorly
                # (see bench/costmodel_sweep.py), so every schedule is timed
                best = [tuned(Gs[0], fb, ref, f"jacobi {c3}^3", full=True)]
            else:
                # tune each colour on its own (in place: every candidate starts from the same input)
                r0 = stencil_ref("rbsor", f["pressure"], f["rhs"], f["cellType"], dims, 0, 1.6)
                best = [tuned(Gs[0], fb, r0, f"rbsor-red {c3}^3", full=True)]
                best.append(dict(best[0]))  # same schedule for black (identical access pattern)
            est = TUNE_LOG[f"jacobi {c3}^3" if kind == "jacobi" else f"rbsor-red {c3}^3"]["best_s"] * len(Gs)
            R = repeat_for(est)
            job = GpuJob()
            for name, b in fb.items():
                job.buffer(name, b)
            job.buffer("P", baselines.flipgpu_params(dims, 1.6))
            job.buffer("pressureOut", 4 * nodes)
            outname = Gs[0].output.name
            if kind == "jacobi":
                job.buffer(outname, 4 * nodes)
            hp, hs = baselines.flipgpu_case(kind, dims)
            job.case("hand", hp, hs, readback=["pressureOut" if kind == "jacobi" else "pressure"], repeat=R, runs=RUNS)
            for cid, kns in (("gen_default", [None] * len(Gs)), ("gen_tuned", best)):
                cks = [compile_group(G, T, kn) for G, kn in zip(Gs, kns)]
                job.case(cid, [{"name": f"k{i}", "code": ck.source, "entry": ck.kernel.name} for i, ck in enumerate(cks)],
                         [{"pipeline": f"k{i}", "bindings": ck.params, "groups": list(ck.grid)} for i, ck in enumerate(cks)],
                         readback=[cks[-1].output], repeat=R, runs=RUNS)
            res, adapter = finish(job)
            shp = f["pressure"].shape
            hname = "pressureOut" if kind == "jacobi" else "pressure"
            gname = outname if kind == "jacobi" else "pressure"
            errs = {"hand": rel_err(np.frombuffer(res["hand"]["data"][hname], np.float32).reshape(shp), ref)}
            for cid in ("gen_default", "gen_tuned"):
                errs[cid] = rel_err(np.frombuffer(res[cid]["data"][gname], np.float32).reshape(shp), ref)
            label = f"{'Jacobi step' if kind == 'jacobi' else 'RB-SOR iter'} {c3}^3"
            rows_out.append(row("pressure", label, [c3, c3, c3], "f32", "Gnode/s", float(nodes), res, errs,
                                best[0], {"bytes_per_node": 16 if kind == "jacobi" else 28}))
    return rows_out, adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("-o", default=str(ROOT / "bench/results/rtx2060.json"))
    a = ap.parse_args()
    fams = a.only.split(",") if a.only else ["matvec", "ffn", "gemm", "pressure"]
    t0 = time.time()
    rows, adapter = [], None
    if "matvec" in fams:
        sh = MATVEC_SHAPES[1:2] if a.quick else MATVEC_SHAPES
        r, adapter = bench_matvec(sh, ["f32", "q8_0", "q4_0"])
        rows += r
    if "ffn" in fams:
        r, adapter = bench_ffn([MATVEC_SHAPES[1]] if a.quick else [MATVEC_SHAPES[1], MATVEC_SHAPES[3]])
        rows += r
    if "gemm" in fams:
        cs = [("prefill 128 tok", 128, 1536, 1536, "f32")]
        if not a.quick:
            cs += [("prefill 128 tok", 128, 1536, 1536, "q8_0"), ("square 1024", 1024, 1024, 1024, "f32")]
        r, adapter = bench_gemm(cs)
        rows += r
    if "pressure" in fams:
        r, adapter = bench_pressure([64] if a.quick else [64, 128])
        rows += r
    meta = {"date": dt.date.today().isoformat(), "adapter": adapter, "host": platform.platform(),
            "cpu": platform.processor(), "python": platform.python_version(), "runs": RUNS,
            "timer": "WebGPU timestamp-query (per compute pass, divided by repeat count)",
            "webgpu": "Dawn via the `webgpu` npm package 0.6.2 (D3D12 backend)", "seconds": round(time.time() - t0),
            "other_gpu_load_percent_before_each_job": UTIL}
    out = Path(a.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    tl = out.with_name(out.stem + "_tuning.json")
    tl.write_text(json.dumps(TUNE_LOG, indent=1))
    print(f"wrote {out} and {tl} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
