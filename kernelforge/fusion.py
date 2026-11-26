"""Operator fusion: partition the graph into kernels (TVM-style pattern rules).

Pattern kinds of a group:
  injective  chains of elementwise / broadcast ops                     -> one flat kernel
  matmul     matmul/linear anchor + injective prologues on its operands + elementwise epilogue
  row        reductions over the last axis + the elementwise ops around them that share the same
             rows (RMSNorm, softmax, mean/var ...)                      -> one workgroup per row
  stencil    flipgpu-style 3-D stencil anchor + injective prologue/epilogue

Rules (applied in topological order):
  * an injective producer is inlined into its consumer's group when nothing else needs it
    materialised (recomputed per element, like ``compute_inline``);
  * an elementwise consumer that reads a matmul/stencil result at the *same* index becomes that
    group's epilogue (applied to the accumulator before the store);
  * an op whose leading (row) shape equals a row group's rows joins that row group.
After partitioning, any node that is consumed outside its group but is not the group's output is
forced to materialise and fusion is re-run, so every group has exactly one output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import ANCHOR, BINARY, REDUCE, SCALAR, STENCIL, UNARY, Graph, Node

INJECTIVE = UNARY + BINARY + SCALAR


def kind_of(op: str) -> str:
    if op in INJECTIVE: return "injective"
    if op in REDUCE: return "row"
    if op in ANCHOR: return "matmul"
    if op in STENCIL: return "stencil"
    raise ValueError(op)


@dataclass(eq=False)
class Group:
    nodes: list = field(default_factory=list)
    kind: str = "injective"
    anchor: Node | None = None
    rows: tuple | None = None  # row groups: leading shape

    @property
    def output(self) -> Node:
        return self.nodes[-1]

    def __repr__(self):
        return f"<{self.kind} group: {', '.join(n.name + ':' + n.op for n in self.nodes)}>"


def _partition(g: Graph, forced: set, enable: bool) -> list[Group]:
    cons = g.consumers()
    outs = set(g.outputs) | forced
    group_of: dict = {}
    groups: list[Group] = []

    def inlinable(i: Node, n: Node) -> bool:
        if i.op == "input" or i in outs:
            return False
        P = group_of[i]
        return P.kind == "injective" and cons[i] == [n] and P.output is i

    for n in g.nodes:
        if n.op == "input":
            continue
        k = kind_of(n.op)
        G = Group([n], k, n if k in ("matmul", "stencil") else None,
                  n.inputs[0].shape[:-1] if k == "row" else None)
        if enable:
            host = None
            for i in n.inputs:
                if i.op == "input" or i in outs:
                    continue
                P = group_of[i]
                if P.output is not i or P is host:
                    continue
                if k == "injective" and P.kind in ("matmul", "stencil") and i.shape == n.shape \
                        and cons[i] == [n] and host is None:
                    host = P
                elif k in ("injective", "row") and P.kind == "row" and host is None \
                        and n.shape[:-1] == P.rows and len(n.shape) == len(P.rows) + 1 \
                        and all(c.op in INJECTIVE + REDUCE for c in cons[i]):
                    host = P
            absorbed = []
            for i in n.inputs:
                if i.op != "input" and inlinable(i, n) and group_of[i] is not host:
                    if group_of[i] not in absorbed:
                        absorbed.append(group_of[i])
            if host is not None:
                for P in absorbed:
                    host.nodes.extend(P.nodes)
                    groups.remove(P)
                    for m in P.nodes:
                        group_of[m] = host
                host.nodes.append(n)
                group_of[n] = host
                # keep topological order inside the group
                order = {m: j for j, m in enumerate(g.nodes)}
                host.nodes.sort(key=order.get)
                continue
            for P in absorbed:
                G.nodes[:0] = P.nodes
                groups.remove(P)
                for m in P.nodes:
                    group_of[m] = G
            order = {m: j for j, m in enumerate(g.nodes)}
            G.nodes.sort(key=order.get)
        groups.append(G)
        for m in G.nodes:
            group_of[m] = G
    return groups


def fuse(g: Graph, enable: bool = True) -> list[Group]:
    forced: set = set()
    while True:
        groups = _partition(g, forced, enable)
        cons = g.consumers()
        group_of = {m: G for G in groups for m in G.nodes}
        bad = set()
        for G in groups:
            for m in G.nodes:
                external = any(group_of[c] is not G for c in cons[m]) or m in g.outputs
                if external and m is not G.output:
                    bad.add(m)
        if not bad:
            return groups
        forced |= bad
