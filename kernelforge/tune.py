"""Small autotuner: grid search over a template's knob space, pruned by the cost model, measured on
the GPU (all candidates of one kernel are timed in a single harness process on the same buffers).

Every measured candidate's output is also checked against the NumPy oracle; a candidate that
compiles but computes the wrong answer is rejected (and would be a compiler bug: the test suite
runs the same check on every template).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .compiler import compile_group, form_for
from .compute import compute_def
from .cost import predict
from .fusion import fuse
from .graph import Graph, evaluate
from .runtime import GpuJob, to_bytes
from .schedule import ScheduleError, default_knobs, space, template_for
from .target import Target


@dataclass
class TuneResult:
    template: str
    best: dict
    best_s: float
    default_s: float
    log: list = field(default_factory=list)  # [{knobs, pred_s, meas_s, ok}]
    seconds: float = 0.0
    space_size: int = 0


def group_feeds(g: Graph, feeds: dict):
    vals = evaluate(g, feeds)
    return vals


def measure(cks: list, feeds_bytes: dict, out_name: str, out_shape, ref: np.ndarray, repeat: int,
            runs: int, backend: str = "wgsl", rtol: float = 2e-3) -> list:
    """Time compiled kernels (one case each) on shared buffers. Returns [(seconds|None, ok, err)]."""
    job = GpuJob(backend)
    try:
        sizes = {}
        for ck in cks:
            for p in ck.kernel.params:
                from .passes import storage_size
                sizes[p.name] = storage_size(p) * (4 if p.dtype in ("f32", "u32", "i32") else 1)
        for n, nb in sizes.items():
            job.buffer(n, feeds_bytes[n] if n in feeds_bytes else nb)
        for i, ck in enumerate(cks):
            job.case(f"c{i}", [{"name": "p", "code": ck.source, "entry": ck.kernel.name}],
                     [{"pipeline": "p", "bindings": ck.params, "groups": list(ck.grid),
                       "threads": list(ck.workgroup)}],
                     readback=[ck.output], repeat=repeat[i] if isinstance(repeat, list) else repeat,
                     runs=runs, warmup=1)
        res = job.run()
        out = []
        scale = float(np.abs(ref).max()) or 1.0
        for i, c in enumerate(res["cases"]):
            if not c["ok"]:
                out.append((None, False, c.get("error", "")))
                continue
            got = np.frombuffer(c["data"][cks[i].output], np.float32)[: ref.size].reshape(ref.shape)
            err = float(np.abs(got - ref).max()) / scale
            out.append((c["median_ns"] * 1e-9, err < rtol, f"rel.err {err:.2e}"))
        return out
    finally:
        job.cleanup()


def tune_group(G, feeds_bytes: dict, ref: np.ndarray, t: Target, top_k: int = 16, full: bool = False,
               repeat: int = 0, runs: int = 15, log=print, backend: str = "wgsl") -> TuneResult:
    t0 = time.time()
    cd = compute_def(G, form_for(G))
    tmpl = template_for(cd)
    cands = space(tmpl, cd, t)
    scored = sorted(((predict(tmpl, cd, kn, t), kn) for kn in cands), key=lambda x: x[0])
    chosen = scored if full else scored[:top_k]
    dflt = default_knobs(tmpl)
    entries = [(predict(tmpl, cd, dflt, t), dflt)] + chosen
    cks, keep = [], []
    for pred, kn in entries:
        try:
            cks.append(compile_group(G, t, kn))
            keep.append((pred, kn))
        except ScheduleError as e:
            log(f"  skip {kn}: {e}")
    results = []
    B = 48  # candidates per harness process
    for s in range(0, len(cks), B):
        # the SAME, small repetition count for every candidate: short passes (a few hundred us)
        # are less exposed to time-slicing with other GPU work than long ones
        reps = [repeat or 3 for _ in keep[s:s + B]]
        results += measure(cks[s:s + B], feeds_bytes, G.output.name, G.output.shape, ref, reps, runs, backend)
    logrows = []
    for (pred, kn), (sec, ok, msg) in zip(keep, results):
        logrows.append({"knobs": kn, "pred_s": pred, "meas_s": sec, "ok": ok, "note": msg})
    good = [r for r in logrows if r["ok"] and r["meas_s"] is not None]  # the default competes too
    bad = [r for r in logrows if not r["ok"]]
    if bad:
        log(f"  WARNING: {len(bad)} candidate(s) rejected: {[ (r['knobs'], r['note']) for r in bad[:3]]}")
    best = min(good, key=lambda r: r["meas_s"]) if good else logrows[0]
    return TuneResult(tmpl, best["knobs"], best["meas_s"], logrows[0]["meas_s"], logrows,
                      time.time() - t0, len(cands))


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def feeds_to_bytes(g: Graph, feeds: dict) -> dict:
    return {n.name: to_bytes(feeds[n.name], n.dtype) for n in g.inputs()}


def single_group(g: Graph):
    groups = fuse(g)
    assert len(groups) == 1, groups
    return groups[0]
