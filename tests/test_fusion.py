from kernelforge import zoo
from kernelforge.fusion import fuse
from kernelforge.graph import Graph


def kinds(g):
    return [(G.kind, [n.op for n in G.nodes]) for G in fuse(g)]


def test_rmsnorm_is_one_row_kernel():
    k = kinds(zoo.rmsnorm(4, 64))
    assert k == [("row", ["mul", "mean", "adds", "rsqrt", "mul", "mul"])]


def test_softmax_two_reductions_one_kernel():
    assert [x[0] for x in kinds(zoo.softmax(4, 50))] == ["row"]


def test_matvec_epilogue_fusion():
    k = kinds(zoo.ffn_gate(64, 64))
    assert k == [("matmul", ["linear"]), ("matmul", ["linear", "silu", "mul"])]


def test_elementwise_chain_and_broadcast():
    assert len(fuse(zoo.elementwise())) == 1


def test_no_fusion_flag():
    assert len(fuse(zoo.rmsnorm(4, 64), enable=False)) == 6


def test_shared_intermediate_is_materialised():
    g = Graph()
    x = g.input("x", (16,))
    y = g.exp(x)
    g.output(y * 2.0, y + 1.0)
    groups = fuse(g)
    outs = {G.output.op for G in groups}
    assert "exp" in outs and len(groups) == 3


def test_mlp_partition():
    k = [x[0] for x in kinds(zoo.chain_mlp())]
    assert k == ["row", "matmul", "matmul"]
