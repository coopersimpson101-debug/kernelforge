"""Every generated kernel vs the NumPy oracle on random inputs: C reference backend, and WGSL on
the GPU (default schedule + sampled tuned schedules from each template's knob space)."""
import random

import numpy as np
import pytest

from conftest import assert_close
from kernelforge import zoo
from kernelforge.compiler import compile_graph
from kernelforge.compute import compute_def
from kernelforge.compiler import form_for
from kernelforge.fusion import fuse
from kernelforge.graph import evaluate, random_feeds
from kernelforge.runtime import as_f32, run_c, run_wgsl
from kernelforge.schedule import ScheduleError, space, template_for
from kernelforge.target import C_REF, RTX2060_WGSL, WGSL_PORTABLE

NAMES = sorted(zoo.SMALL)
RTOL = {"elementwise": 2e-3, "softmax": 2e-4, "mlp": 2e-3}


def _ref(g, seed):
    f = random_feeds(g, seed)
    vals = evaluate(g, f)
    return f, {o.name: vals[o.name] for o in g.outputs}


@pytest.mark.parametrize("name", NAMES)
def test_c_oracle_matches_numpy(cc, name):
    g = zoo.SMALL[name]()
    f, refs = _ref(g, 1)
    prog = compile_graph(g, C_REF)
    out = run_c(prog, f)
    for o in g.outputs:
        assert_close(as_f32(out[o.name], o.shape), refs[o.name], RTOL.get(name, 1e-4), name)


def _sample_knobs(g, t, n, seed):
    """n random schedules per kernel (same index for every kernel of the program)."""
    rnd = random.Random(seed)
    per = []
    for G in fuse(g):
        cd = compute_def(G, form_for(G))
        sp = space(template_for(cd), cd, t)
        rnd.shuffle(sp)
        per.append((template_for(cd), sp[:n]))
    return per


@pytest.mark.parametrize("name", NAMES)
def test_wgsl_default_schedule(gpu, name):
    g = zoo.SMALL[name]()
    f, refs = _ref(g, 2)
    res = run_wgsl(compile_graph(g, RTX2060_WGSL), f)
    for o in g.outputs:
        assert_close(as_f32(res["data"][o.name], o.shape), refs[o.name], RTOL.get(name, 1e-4), name)


@pytest.mark.parametrize("name", NAMES)
def test_wgsl_sampled_schedules(gpu, name):
    """3 random points of every template's knob space (tiling, staging, vectorisation, unroll...)."""
    g = zoo.SMALL[name]()
    f, refs = _ref(g, 3)
    per = _sample_knobs(g, RTX2060_WGSL, 3, seed=hash(name) % 1000)
    for i in range(3):
        picks = {tmpl: (sp[i] if i < len(sp) else None) for tmpl, sp in per}
        try:
            prog = compile_graph(g, RTX2060_WGSL, knobs=lambda cd, tmpl: picks.get(tmpl))
        except ScheduleError:
            continue
        res = run_wgsl(prog, f)
        for o in g.outputs:
            assert_close(as_f32(res["data"][o.name], o.shape), refs[o.name], RTOL.get(name, 1e-4),
                         f"{name} {picks}")


@pytest.mark.parametrize("name", ["matvec_q4_0", "softmax", "gemm", "jacobi"])
def test_wgsl_portable_limits(gpu, name):
    """Without subgroups and with WebGPU's default 256-thread / 16 KiB limits."""
    g = zoo.SMALL[name]()
    f, refs = _ref(g, 4)
    res = run_wgsl(compile_graph(g, WGSL_PORTABLE), f)
    for o in g.outputs:
        assert_close(as_f32(res["data"][o.name], o.shape), refs[o.name], RTOL.get(name, 1e-4), name)
