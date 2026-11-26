"""Driver: graph -> fused groups -> compute defs -> scheduled loop nests -> passes -> source text."""
from __future__ import annotations

from dataclasses import dataclass, field

from .codegen.c import emit_c
from .codegen.msl import emit_msl
from .codegen.wgsl import emit_wgsl
from .compute import ComputeDef, compute_def
from .fusion import Group, fuse
from .graph import Graph
from .ir import Kernel
from .passes import finalize, storage_size
from .schedule import default_knobs, schedule, template_for
from .target import C_REF, Target


@dataclass
class CompiledKernel:
    name: str
    source: str
    target: str
    kernel: Kernel
    cd: ComputeDef
    group: Group
    params: list  # tensor names in binding order (aliased output -> its input)
    output: str
    workgroup: tuple
    grid: tuple
    knobs: dict
    template: str


@dataclass
class Program:
    graph: Graph
    target: Target
    kernels: list = field(default_factory=list)

    def tensors(self) -> dict:
        """name -> (storage bytes, dtype) for every buffer the program touches."""
        out = {}
        for ck in self.kernels:
            for p in ck.kernel.params:
                out[p.name] = (storage_size(p) * (4 if p.dtype in ("f32", "u32", "i32") else 1), p.dtype)
        return out

    def source(self) -> str:
        return "\n".join(ck.source for ck in self.kernels)


def form_for(G: Group) -> str:
    """Quantised linear layers use the block-factored compute (scale applied per 32-block, like
    tinyinfer/ggml) in the matvec template; GEMM dequantises on load into shared memory."""
    if G.kind == "matmul" and G.anchor.op == "linear":
        x = G.anchor.inputs[0]
        m = 1 if len(x.shape) == 1 else x.shape[0]
        quantised = G.anchor.inputs[1].dtype in ("q8_0", "q4_0")
        return "blocked" if (m == 1 and quantised) else "flat"
    return "flat"


def emit(k: Kernel, t: Target) -> str:
    return {"wgsl": emit_wgsl, "msl": emit_msl, "c": emit_c}[t.name](k)


def compile_group(G: Group, t: Target, knobs: dict | None = None) -> CompiledKernel:
    cd = compute_def(G, form_for(G))
    k = schedule(cd, knobs, t)
    k = finalize(k, t.name)
    src = emit(k, t)
    params = [k.alias.get(p.name, p.name) for p in k.params]
    tmpl = "cpu" if not t.gpu else template_for(cd)
    return CompiledKernel(k.name, src, t.name, k, cd, G, params, k.alias.get(cd.out.name, cd.out.name),
                          k.workgroup, k.grid, k.meta.get("knobs", {}), tmpl)


def compile_graph(g: Graph, t: Target = C_REF, knobs=None, fusion: bool = True) -> Program:
    """``knobs``: None (default schedules), a dict {template: knobs}, or a callable(cd, tmpl) -> knobs."""
    prog = Program(g, t)
    for G in fuse(g, enable=fusion):
        kn = None
        if t.gpu and knobs is not None:
            cd = compute_def(G, form_for(G))
            tmpl = template_for(cd)
            kn = knobs(cd, tmpl) if callable(knobs) else knobs.get(tmpl)
        prog.kernels.append(compile_group(G, t, kn))
    return prog


__all__ = ["compile_graph", "compile_group", "Program", "CompiledKernel", "default_knobs"]
