"""Knob spaces, cost model and tuner plumbing (compile-only; GPU measurement is covered elsewhere)."""
import random

import pytest

from kernelforge import zoo
from kernelforge.compiler import compile_group, form_for
from kernelforge.compute import compute_def
from kernelforge.cost import predict
from kernelforge.fusion import fuse
from kernelforge.schedule import ScheduleError, space, template_for
from kernelforge.target import M1_MSL, RTX2060_WGSL, WGSL_PORTABLE
from kernelforge.tune import spearman


@pytest.mark.parametrize("name", ["matvec_q4_0", "matvec_f32", "gemm", "jacobi", "softmax", "elementwise"])
@pytest.mark.parametrize("t", [RTX2060_WGSL, WGSL_PORTABLE, M1_MSL], ids=lambda t: t.device)
def test_space_points_compile_and_cost(name, t):
    G = fuse(zoo.SMALL[name]())[0]
    cd = compute_def(G, form_for(G))
    tmpl = template_for(cd)
    sp = space(tmpl, cd, t)
    assert sp, "empty knob space"
    ok = 0
    for kn in random.Random(0).sample(sp, min(6, len(sp))):
        assert predict(tmpl, cd, kn, t) > 0
        try:
            ck = compile_group(G, t, kn)
        except ScheduleError:
            continue  # e.g. shared-memory limit: the tuner skips these too
        assert ck.kernel.threads <= t.max_threads
        ok += 1
    assert ok > 0


def test_cost_model_prefers_coalesced_matvec():
    G = fuse(zoo.matvec(4096, 4096, "f32"))[0]
    cd = compute_def(G, form_for(G))
    naive = {"R": 64, "T": 1, "U": 1, "V": 1, "xlane": "shared", "stage_x": 0}
    good = {"R": 2, "T": 128, "U": 2, "V": 4, "xlane": "shared", "stage_x": 0}
    assert predict("matvec", cd, good, RTX2060_WGSL) < predict("matvec", cd, naive, RTX2060_WGSL)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
