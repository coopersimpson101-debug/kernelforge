import functools
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernelforge import runtime  # noqa: E402


@functools.lru_cache(maxsize=None)
def gpu_skip_reason() -> str:
    """'' if WGSL kernels can run here (Node + Dawn + a WebGPU adapter), else why not."""
    if os.environ.get("KF_NO_GPU") == "1":
        return "KF_NO_GPU=1"
    if os.environ.get("CI") and os.environ.get("KF_GPU") != "1":
        return "hosted CI runner without a GPU (set KF_GPU=1 on a GPU runner)"
    if runtime.node_available() is None:
        return "node not found"
    job = runtime.GpuJob()
    try:
        job.buffer("o", 16)
        job.case("probe", [{"name": "p", "entry": "main",
                            "code": "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
                                    "@compute @workgroup_size(1) fn main() { o[0] = 1.0; }"}],
                 [{"pipeline": "p", "bindings": ["o"], "groups": [1, 1, 1]}], readback=["o"])
        res = job.run(timeout=300)
        if not res["cases"][0]["ok"]:
            return "probe kernel failed: " + res["cases"][0].get("error", "")
        return ""
    except Exception as e:  # noqa: BLE001
        return f"WebGPU unavailable: {e}"[:300]
    finally:
        job.cleanup()


@functools.lru_cache(maxsize=None)
def cc_skip_reason() -> str:
    return "" if runtime.find_cc() else "no C compiler (gcc/clang or pip install ziglang)"


@pytest.fixture
def gpu():
    r = gpu_skip_reason()
    if r:
        pytest.skip(r)


@pytest.fixture
def cc():
    r = cc_skip_reason()
    if r:
        pytest.skip(r)


def assert_close(got, ref, rtol, what=""):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    assert got.shape == ref.shape
    assert np.all(np.isfinite(got)), f"{what}: non-finite output"
    scale = max(float(np.abs(ref).max()), 1e-6)
    err = float(np.abs(got - ref).max()) / scale
    assert err < rtol, f"{what}: max rel. error {err:.3e} >= {rtol}"
    return err
