"""kernelforge command line.

    python -m kernelforge list
    python -m kernelforge explain ffn_q4_0              # graph -> groups -> compute -> loop nest -> WGSL
    python -m kernelforge compile matvec_q4_0 --target msl [--knobs '{"R": 8, "T": 32}']
    python -m kernelforge run softmax                   # C oracle + WebGPU vs NumPy on random inputs
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import zoo
from .analysis import simplify_kernel
from .compiler import compile_graph, form_for
from .compute import compute_def
from .fusion import fuse
from .graph import evaluate, random_feeds
from .ir import show_stmt
from .schedule import schedule, template_for
from .target import TARGETS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kernelforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("explain", "compile", "run"):
        p = sub.add_parser(name)
        p.add_argument("graph", choices=sorted(zoo.SMALL))
        p.add_argument("--target", default="rtx2060", choices=sorted(TARGETS))
        p.add_argument("--knobs", default=None, help="JSON knob dict applied to every kernel's template")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n in sorted(zoo.SMALL):
            print(n)
        return 0
    g = zoo.SMALL[a.graph]()
    t = TARGETS[a.target]
    kn = json.loads(a.knobs) if a.knobs else None
    knobs = (lambda cd, tmpl: kn) if kn else None
    if a.cmd == "explain":
        print(g, "\n")
        for G in fuse(g):
            cd = compute_def(G, form_for(G))
            print(f"== group {G}\n{cd}\n")
            k = schedule(cd, kn, t)
            print(f"-- loop nest ({template_for(cd)} template, knobs {k.meta['knobs']}) before passes:")
            print(show_stmt(simplify_kernel(k).body), "\n")
        prog = compile_graph(g, t, knobs)
        for ck in prog.kernels:
            print(f"-- {ck.target} for {ck.name}:\n{ck.source}")
        return 0
    if a.cmd == "compile":
        print(compile_graph(g, t, knobs).source())
        return 0
    from .runtime import as_f32, run_c, run_wgsl
    f = random_feeds(g, 0)
    ref = evaluate(g, f)
    outs = {"c": run_c(compile_graph(g, TARGETS["c"]), f)}
    try:
        outs["wgsl"] = run_wgsl(compile_graph(g, t if t.name == "wgsl" else TARGETS["rtx2060"], knobs), f)["data"]
    except Exception as e:  # noqa: BLE001
        print(f"wgsl: skipped ({e})")
    for be, data in outs.items():
        for o in g.outputs:
            got = as_f32(data[o.name], o.shape)
            err = np.abs(got - ref[o.name]).max() / max(np.abs(ref[o.name]).max(), 1e-12)
            print(f"{be:5s} {o.name}: max rel. error vs NumPy {err:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
