"""C99 reference backend: the serial loop nest of ``cpu_schedule``, used as a correctness oracle
(compiled with the system C compiler or ``python -m ziglang cc``)."""
from __future__ import annotations

from ..ir import (Bin, ByteLoad, Call, Cast, Const, Decl, Expr, For, HalfLoad, Kernel, Load, Select, Var,
                  WordLoad, elem, lanes)
from .base import TextGen, ident

_T = {"f32": "float", "i32": "int32_t", "u32": "uint32_t", "bool": "int"}

PRELUDE = r"""#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(_WIN32)
#define KF_EXPORT __declspec(dllexport)
#else
#define KF_EXPORT
#endif
#define KF_MIN(a, b) ((a) < (b) ? (a) : (b))
#define KF_MAX(a, b) ((a) > (b) ? (a) : (b))
static inline float kf_h2f(const uint8_t *p) {  /* IEEE half -> float, exact */
    uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int k = -1; do { k++; m <<= 1; } while (!(m & 0x400)); m &= 0x3ff;
               bits = s | ((uint32_t)(127 - 15 - k) << 23) | (m << 13); }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline uint32_t kf_ld32(const uint8_t *p) { uint32_t w; memcpy(&w, p, 4); return w; }
"""


def flit(v: float) -> str:
    r = repr(float(v))
    if "e" not in r and "." not in r:
        r += ".0"
    return r + "f"


class CGen(TextGen):
    indent_unit = "    "

    def e(self, x: Expr) -> str:
        if isinstance(x, Const):
            if x.dtype == "f32": return flit(x.value)
            if x.dtype == "u32": return f"{int(x.value)}u"
            if x.dtype == "bool": return "1" if x.value else "0"
            return str(int(x.value))
        if isinstance(x, Var):
            return x.name
        if isinstance(x, Bin):
            if x.op in ("min", "max"):
                if elem(x.dtype) == "f32":
                    return f"f{x.op}f({self.e(x.a)}, {self.e(x.b)})"
                return f"KF_{x.op.upper()}({self.e(x.a)}, {self.e(x.b)})"
            return f"({self.e(x.a)} {x.op} {self.e(x.b)})"
        if isinstance(x, Select):
            return f"({self.e(x.cond)} ? {self.e(x.t)} : {self.e(x.f)})"
        if isinstance(x, Cast):
            return f"(({_T[x.dtype]})({self.e(x.e)}))"
        if isinstance(x, Call):
            a = [self.e(y) for y in x.args]
            if x.fn == "sext8": return f"((int32_t)(int8_t)({a[0]}))"
            if x.fn == "rsqrt": return f"(1.0f / sqrtf({a[0]}))"
            fn = {"exp": "expf", "sqrt": "sqrtf", "tanh": "tanhf", "abs": "fabsf"}[x.fn]
            return f"{fn}({a[0]})"
        name = ident(x.buf.name)
        if isinstance(x, Load): return f"{name}[{self.e(x.idx)}]"
        if isinstance(x, ByteLoad): return f"((uint32_t){name}[{self.e(x.off)}])"
        if isinstance(x, HalfLoad): return f"kf_h2f({name} + {self.e(x.off)})"
        if isinstance(x, WordLoad): return f"kf_ld32({name} + {self.e(x.off)})"
        raise NotImplementedError(f"c: {x!r}")

    def for_head(self, s: For) -> str:
        v = s.var.name
        return f"for (int32_t {v} = 0; {v} < {s.extent}; {v}++) {{"

    def decl(self, s: Decl) -> str:
        kw = "" if s.mutable else "const "
        return f"{kw}{_T[s.var.dtype]} {s.var.name} = {self.e(s.init)};"

    def barrier(self) -> str:
        raise AssertionError("C oracle kernels are serial")


def c_signature(k: Kernel) -> list:
    out = []
    for p in k.params:
        b = k.buffers[k.alias.get(p.name, p.name)]
        el = {"f32": "float", "u32": "uint32_t", "i32": "int32_t", "bytes": "uint8_t"}[b.dtype]
        q = "const " if b.readonly else ""
        out.append((b, f"{q}{el} *{ident(b.name)}"))
    return out


def emit_c(k: Kernel, prelude: bool = True) -> str:
    g = CGen(k)
    for b in k.allocs:
        g.line(f"{_T[b.dtype]} {ident(b.name)}[{b.size}];")
    g.stmt(k.body)
    sig = ", ".join(s for _, s in c_signature(k))
    head = PRELUDE if prelude else ""
    return head + f"\nKF_EXPORT void {k.name}({sig}) {{\n" + "\n".join(g.lines) + "\n}\n"
