"""Execution back-ends: the C oracle (compiled shared library via ctypes) and WebGPU (WGSL via the
Node/Dawn harness in ``harness/run.ts``)."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .codegen.c import PRELUDE, c_signature, emit_c
from .compiler import Program

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness" / "run.ts"
METAL_RUNNER = ROOT / "build" / "kf_metal_run"


# ----------------------------------------------------------------------------- host-side buffers

def to_bytes(arr: np.ndarray, dtype: str) -> bytes:
    if dtype in ("q8_0", "q4_0"):
        return np.ascontiguousarray(arr, np.uint8).tobytes()
    if dtype == "f16":
        return np.ascontiguousarray(arr, np.float16).tobytes()
    if dtype == "u32":
        return np.ascontiguousarray(arr, np.uint32).tobytes()
    return np.ascontiguousarray(arr, np.float32).tobytes()


def initial_buffers(prog: Program, feeds: dict) -> dict:
    """name -> bytearray for every tensor the program touches (inputs filled, others zero)."""
    out = {}
    for name, (nbytes, dt) in prog.tensors().items():
        if name in feeds:
            b = bytearray(to_bytes(feeds[name], dt))
            assert len(b) == nbytes, (name, len(b), nbytes)
        else:
            b = bytearray(nbytes)
        out[name] = b
    return out


# ----------------------------------------------------------------------------- C oracle

def find_cc() -> list[str] | None:
    if os.environ.get("KF_CC"):
        return os.environ["KF_CC"].split()
    for c in ("cc", "gcc", "clang"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "cc"]
    except ImportError:
        return None


_libs: dict = {}


def build_c(prog: Program) -> ctypes.CDLL:
    src = PRELUDE + "\n".join(emit_c(ck.kernel, prelude=False) for ck in prog.kernels)
    key = hashlib.sha1(src.encode()).hexdigest()[:16]
    if key in _libs:
        return _libs[key]
    cc = find_cc()
    if cc is None:
        raise RuntimeError("no C compiler (set KF_CC, install gcc/clang, or pip install ziglang)")
    d = Path(os.environ.get("KF_CACHE", Path(tempfile.gettempdir()) / "kernelforge-cache"))
    d.mkdir(parents=True, exist_ok=True)
    ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    lib = d / f"kf_{key}{ext}"
    if not lib.exists():
        cfile = d / f"kf_{key}.c"
        cfile.write_text(src)
        cmd = cc + ["-O1", "-shared", "-fPIC", "-ffp-contract=off", "-o", str(lib), str(cfile)]
        if os.name != "nt":
            cmd.append("-lm")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"C compile failed:\n{r.stderr}\n{src[:4000]}")
    _libs[key] = ctypes.CDLL(str(lib))
    return _libs[key]


def run_c(prog: Program, feeds: dict) -> dict:
    lib = build_c(prog)
    bufs = initial_buffers(prog, feeds)
    cbufs = {n: (ctypes.c_uint8 * len(b)).from_buffer(b) for n, b in bufs.items()}
    for ck in prog.kernels:
        fn = getattr(lib, ck.kernel.name)
        args = [ctypes.cast(cbufs[b.name], ctypes.c_void_p) for b, _ in c_signature(ck.kernel)]
        fn.argtypes = [ctypes.c_void_p] * len(args)
        fn(*args)
    out = {n: bytes(b) for n, b in bufs.items()}
    for ck in prog.kernels:  # in-place kernels: expose the result under the graph node's name too
        out.setdefault(ck.cd.out.name, out[ck.output])
    return out


def as_f32(raw: bytes, shape) -> np.ndarray:
    return np.frombuffer(raw, np.float32)[: int(np.prod(shape))].reshape(shape)


# ----------------------------------------------------------------------------- WebGPU harness

def node_available() -> str | None:
    return shutil.which("node")


class GpuJob:
    """Collects buffers + cases, runs them in ONE harness process (one device, one upload)."""

    def __init__(self, backend: str = "wgsl"):
        self.backend = backend
        self.dir = Path(tempfile.mkdtemp(prefix="kfjob-"))
        self.buffers: dict = {}
        self.cases: list = []

    def buffer(self, name: str, data: bytes | int):
        if isinstance(data, int):
            self.buffers[name] = {"bytes": data}
        else:
            f = f"{name}.in.bin"
            (self.dir / f).write_bytes(bytes(data))
            self.buffers[name] = {"bytes": len(data), "init": f}

    def case(self, cid: str, pipelines: list, steps: list, readback=(), repeat: int = 0, runs: int = 0,
             warmup: int = 2):
        self.cases.append({"id": cid, "pipelines": pipelines, "steps": steps, "readback": list(readback),
                           "repeat": repeat, "runs": runs, "warmup": warmup})

    def run(self, timeout: int = 1800) -> dict:
        jp = self.dir / "job.json"
        jp.write_text(json.dumps({"buffers": self.buffers, "cases": self.cases}))
        if self.backend == "msl":
            if not METAL_RUNNER.exists():
                raise RuntimeError(f"{METAL_RUNNER} not built (see metal/kf_metal_run.m)")
            cmd = [str(METAL_RUNNER), str(jp)]
        else:
            node = node_available()
            if node is None:
                raise RuntimeError("node not found")
            cmd = [node, "--no-warnings", str(HARNESS), str(jp)]
        for attempt in range(6):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
            # a shared GPU can be momentarily out of memory for a new device: back off and retry
            if r.returncode and "E_OUTOFMEMORY" in r.stderr and attempt < 5:
                import time
                time.sleep(15 * (attempt + 1))
                continue
            break
        if r.returncode:
            raise RuntimeError(f"harness failed ({r.returncode}):\n{r.stderr[-4000:]}")
        res = json.loads(r.stdout)
        for c in res["cases"]:
            c["data"] = {k: (self.dir / v).read_bytes() for k, v in c.get("outputs", {}).items()}
        return res

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def program_steps(prog: Program, prefix: str = "") -> tuple[list, list]:
    pipes, steps = [], []
    for i, ck in enumerate(prog.kernels):
        pn = f"{prefix}p{i}"
        pipes.append({"name": pn, "code": ck.source, "entry": ck.kernel.name})
        steps.append({"pipeline": pn, "bindings": list(ck.params), "groups": list(ck.grid),
                      "threads": list(ck.workgroup)})
    return pipes, steps


def run_wgsl(prog: Program, feeds: dict, repeat: int = 0, runs: int = 0) -> dict:
    return run_gpu(prog, feeds, repeat, runs, "wgsl")


def run_msl(prog: Program, feeds: dict, repeat: int = 0, runs: int = 0) -> dict:
    return run_gpu(prog, feeds, repeat, runs, "msl")


def run_gpu(prog: Program, feeds: dict, repeat: int = 0, runs: int = 0, backend: str = "wgsl") -> dict:
    job = GpuJob(backend)
    try:
        for n, b in initial_buffers(prog, feeds).items():
            job.buffer(n, bytes(b) if n in feeds else len(b))
        pipes, steps = program_steps(prog)
        outs = [ck.output for ck in prog.kernels]
        job.case("prog", pipes, steps, readback=outs, repeat=repeat, runs=runs)
        res = job.run()
        c = res["cases"][0]
        if not c["ok"]:
            raise RuntimeError(c["error"])
        if res["errors"]:
            raise RuntimeError("; ".join(res["errors"]))
        c["adapter"] = res["adapter"]
        for ck in prog.kernels:
            if ck.output in c["data"]:
                c["data"].setdefault(ck.cd.out.name, c["data"][ck.output])
        return c
    finally:
        job.cleanup()
