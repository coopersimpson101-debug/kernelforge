"""The hand-written baselines must themselves be right before we race against them."""
import numpy as np
import pytest

from conftest import assert_close
from kernelforge import baselines, zoo
from kernelforge.graph import evaluate, random_feeds, stencil_ref
from kernelforge.runtime import GpuJob, to_bytes


def _run(bufs: dict, outs: dict, pipes, steps, readback):
    job = GpuJob()
    try:
        for n, b in bufs.items():
            job.buffer(n, b)
        for n, nb in outs.items():
            job.buffer(n, nb)
        job.case("hw", pipes, steps, readback=readback)
        res = job.run()
        c = res["cases"][0]
        assert c["ok"], c.get("error")
        assert not res["errors"], res["errors"]
        return c["data"]
    finally:
        job.cleanup()


@pytest.mark.parametrize("dtype", ["f32", "q8_0", "q4_0"])
def test_tinyinfer_matvec_port(gpu, dtype):
    rows, cols = 203, 512  # rows not a multiple of 8: exercises the early return
    g = zoo.matvec(rows, cols, dtype)
    f = random_feeds(g, 7)
    ref = evaluate(g, f)[g.outputs[0].name]
    pipes, steps, extra = baselines.tinyinfer_matvec_case(dtype, rows, cols)
    bufs = {"W": to_bytes(f["W"], dtype), "x": to_bytes(f["x"], "f32"), **extra}
    out = _run(bufs, {"y": 4 * rows}, pipes, steps, ["y"])
    assert_close(np.frombuffer(out["y"], np.float32), ref, 1e-4, dtype)


def test_tinyinfer_silu_mul_port(gpu):
    n = 1000
    rng = np.random.default_rng(0)
    gv, u = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    pipes, steps, extra = baselines.tinyinfer_silu_mul_case(n, "g", "u")
    out = _run({"g": baselines.f32(gv), "u": baselines.f32(u), **extra}, {}, pipes, steps, ["g"])
    ref = gv / (1 + np.exp(-gv.astype(np.float64))) * u
    assert_close(np.frombuffer(out["g"], np.float32), ref, 1e-5)


@pytest.mark.parametrize("kind", ["jacobi", "rb"])
def test_flipgpu_pressure_kernels(gpu, kind):
    dims = (10, 12, 14)
    g = zoo.jacobi(nx=14, ny=12, nz=10)
    f = random_feeds(g, 5)
    params = baselines.flipgpu_params(dims, 1.6)
    pipes, steps = baselines.flipgpu_case(kind, dims)
    shape = f["pressure"].shape
    bufs = {"P": params, "pressure": to_bytes(f["pressure"], "f32"), "rhs": to_bytes(f["rhs"], "f32"),
            "cellType": to_bytes(f["cellType"], "u32")}
    out = _run(bufs, {"pressureOut": 4 * f["pressure"].size}, pipes, steps, ["pressure", "pressureOut"])
    if kind == "jacobi":
        ref = stencil_ref("jacobi7", f["pressure"], f["rhs"], f["cellType"], dims)
        got = np.frombuffer(out["pressureOut"], np.float32).reshape(shape)
    else:
        red = stencil_ref("rbsor", f["pressure"], f["rhs"], f["cellType"], dims, 0, 1.6)
        ref = stencil_ref("rbsor", red, f["rhs"], f["cellType"], dims, 1, 1.6)
        got = np.frombuffer(out["pressure"], np.float32).reshape(shape)
    assert_close(got, ref, 1e-5, kind)


@pytest.mark.parametrize("knobs", [None, {"wx": 32, "wy": 4, "wz": 1, "cz": 2, "stage": 0},
                                   {"wx": 16, "wy": 4, "wz": 2, "cz": 1, "stage": 1}])
def test_generated_jacobi_is_bit_identical_to_flipgpu(gpu, knobs):
    """Same arithmetic in the same order: the generated Jacobi step must equal flipgpu's bit for bit."""
    from kernelforge.compiler import compile_graph
    from kernelforge.target import RTX2060_WGSL
    dims = (10, 12, 14)
    g = zoo.jacobi(nx=14, ny=12, nz=10)
    f = random_feeds(g, 9)
    ck = compile_graph(g, RTX2060_WGSL, {"stencil": knobs} if knobs else None).kernels[0]
    pipes, steps = baselines.flipgpu_case("jacobi", dims)
    pipes.append({"name": "gen", "code": ck.source, "entry": ck.kernel.name})
    steps.append({"pipeline": "gen", "bindings": ck.params, "groups": list(ck.grid)})
    bufs = {"P": baselines.flipgpu_params(dims, 1.6), "pressure": to_bytes(f["pressure"], "f32"),
            "rhs": to_bytes(f["rhs"], "f32"), "cellType": to_bytes(f["cellType"], "u32")}
    n = 4 * f["pressure"].size
    out = _run(bufs, {"pressureOut": n, ck.output: n}, pipes, steps, ["pressureOut", ck.output])
    assert out["pressureOut"] == out[ck.output]
