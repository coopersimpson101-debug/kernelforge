"""Shared statement walker for the three text backends."""
from __future__ import annotations

from ..analysis import let_defs, multiple_of
from ..ir import (Assign, Barrier, Bin, Broadcast, ByteLoad, Call, Cast, Comment, Const, Decl, Expr, For,
                  HalfLoad, If, Kernel, Lane, Load, Seq, Select, Stmt, Store, Var, VecLoad, WordLoad, lanes,
                  stmt_exprs, walk)

RESERVED = {"in", "out", "var", "let", "fn", "loop", "select", "array", "ptr", "vec", "mat", "bias",
            "const", "for", "if", "else", "while", "return", "kernel", "device", "constant", "thread",
            "threadgroup", "half", "float", "int", "uint", "bool", "char", "short", "long", "double",
            "signed", "unsigned", "default", "switch", "case", "break", "continue", "struct", "enum",
            "union", "static", "inline", "auto", "register", "restrict", "sizeof", "typedef", "max",
            "min", "dot", "exp", "abs", "main", "target", "meta", "override", "alias", "enable"}


def ident(name: str) -> str:
    return f"{name}_" if name in RESERVED else name


def vec4_buffers(k: Kernel) -> set:
    """Global f32 buffers whose every access is an aligned 4-wide vector: declare as vec4 arrays."""
    seen: dict = {}
    defs = let_defs(k.body)
    for e in stmt_exprs(k.body):
        for x in walk(e):
            if isinstance(x, (Load, VecLoad)) and x.buf.scope == "global":
                ok = isinstance(x, VecLoad) and x.n == 4 and multiple_of(x.idx, defs) % 4 == 0
                seen[x.buf] = seen.get(x.buf, True) and ok
    def stores(s):
        if isinstance(s, Seq):
            for c in s.body:
                yield from stores(c)
        elif isinstance(s, For):
            yield from stores(s.body)
        elif isinstance(s, If):
            yield from stores(s.then)
            if s.els is not None:
                yield from stores(s.els)
        elif isinstance(s, Store):
            yield s
    for s in stores(k.body):
        if s.buf.scope == "global":
            ok = lanes(s.value.dtype) == 4 and multiple_of(s.idx, defs) % 4 == 0
            seen[s.buf] = seen.get(s.buf, True) and ok
    return {b for b, ok in seen.items() if ok and b.dtype == "f32"}


class TextGen:
    indent_unit = "  "

    def __init__(self, k: Kernel):
        self.k = k
        self.lines: list[str] = []
        self.depth = 1
        self.v4 = vec4_buffers(k)
        self.defs = let_defs(k.body)

    def line(self, s: str):
        self.lines.append(self.indent_unit * self.depth + s)

    def stmt(self, s: Stmt):
        if isinstance(s, Seq):
            for c in s.body:
                self.stmt(c)
        elif isinstance(s, For):
            self.line(self.for_head(s))
            self.depth += 1
            self.stmt(s.body)
            self.depth -= 1
            self.line("}")
        elif isinstance(s, If):
            self.line(f"if ({self.e(s.cond)}) {{")
            self.depth += 1
            self.stmt(s.then)
            self.depth -= 1
            if s.els is not None:
                self.line("} else {")
                self.depth += 1
                self.stmt(s.els)
                self.depth -= 1
            self.line("}")
        elif isinstance(s, Decl):
            self.line(self.decl(s))
        elif isinstance(s, Assign):
            self.line(f"{s.var.name} = {self.e(s.value)};")
        elif isinstance(s, Store):
            self.store(s)
        elif isinstance(s, Barrier):
            self.line(self.barrier())
        elif isinstance(s, Comment):
            self.line(f"// {s.text}")
        else:
            raise NotImplementedError(type(s))

    def store(self, s: Store):
        n = lanes(s.value.dtype)
        name = ident(s.buf.name)
        if n == 1:
            self.line(f"{name}[{self.e(s.idx)}] = {self.e(s.value)};")
        elif s.buf in self.v4:
            self.line(f"{name}[{self.e(s.idx)} / 4] = {self.e(s.value)};")
        else:
            self.line("{")
            self.depth += 1
            self.line(self.vec_tmp("vs_", s.value))
            for i in range(n):
                self.line(f"{name}[{self.e(s.idx)} + {i}] = vs_.{'xyzw'[i]};")
            self.depth -= 1
            self.line("}")

    # backend hooks ----------------------------------------------------------
    def e(self, x: Expr) -> str:
        raise NotImplementedError

    def binop(self, x: Bin) -> str:
        if x.op in ("min", "max"):
            return f"{x.op}({self.e(x.a)}, {self.e(x.b)})"
        return f"({self.e(x.a)} {x.op} {self.e(x.b)})"
