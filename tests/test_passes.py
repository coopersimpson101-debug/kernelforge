"""Structural checks of the loop-nest passes on real templates."""
from kernelforge import zoo
from kernelforge.compiler import compile_graph
from kernelforge.ir import WordLoad, stmt_exprs, walk
from kernelforge.target import M1_MSL, RTX2060_WGSL

T = RTX2060_WGSL


def _exprs(ck):
    return [x for e in stmt_exprs(ck.kernel.body) for x in walk(e)]


def test_vectorized_matvec_uses_vec4_buffers():
    p = compile_graph(zoo.matvec(256, 512, "f32"), T,
                      {"matvec": {"R": 2, "T": 32, "U": 2, "V": 4, "xlane": "shared", "stage_x": 0}})
    src = p.kernels[0].source
    assert "array<vec4<f32>>" in src and "dot(" in src


def test_shared_staging_bound_inference_gemm():
    p = compile_graph(zoo.linear_gemm(64, 128, 96), T,
                      {"matmul": {"BM": 32, "BN": 64, "BK": 16, "TM": 4, "TN": 4, "vec": 4, "reg": 1, "pad": 1, "stage": 1}})
    k = p.kernels[0].kernel
    shared = sorted(b.size for b in k.allocs if b.scope == "shared")
    # A tile 32 x (16+1) and W tile 64 x (16+1): footprints inferred from the index expressions
    assert shared == [32 * 17, 64 * 17]
    assert "workgroupBarrier" in p.kernels[0].source


def test_stencil_halo_staging():
    p = compile_graph(zoo.jacobi(30, 30, 30), T, {"stencil": {"wx": 16, "wy": 4, "wz": 2, "cz": 1, "stage": 1}})
    sizes = sorted(b.size for b in p.kernels[0].kernel.allocs)
    assert sizes == [18 * 6 * 4, 18 * 6 * 4]  # (16+2) x (4+2) x (2+2) halo tiles for pressure, cellType


def test_bounds_checks_eliminated_when_divisible():
    p = compile_graph(zoo.matvec(256, 512, "q4_0"), T,
                      {"matvec": {"R": 8, "T": 32, "U": 1, "V": 1, "xlane": "subgroup", "stage_x": 0}})
    assert "select(" not in p.kernels[0].source  # every access proven in range: no guards


def test_q4_bytes_become_word_loads():
    p = compile_graph(zoo.matvec(64, 256, "q4_0"), M1_MSL,
                      {"matvec": {"R": 2, "T": 32, "U": 1, "V": 1, "xlane": "shared", "stage_x": 0, "unroll": 1}})
    assert any(isinstance(x, WordLoad) for x in _exprs(p.kernels[0]))


def test_default_vs_tuned_differ():
    g = zoo.matvec(256, 512, "f32")
    a = compile_graph(g, T).kernels[0].source
    b = compile_graph(g, T, {"matvec": {"R": 2, "T": 32, "U": 2, "V": 4, "xlane": "subgroup",
                                        "stage_x": 1}}).kernels[0].source
    assert a != b and "subgroupAdd" in b
