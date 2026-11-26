"""Metal Shading Language backend.

* everywhere: syntax/type-check the generated MSL as C++ against metal/shim (clang++, g++ or zig c++);
* macOS: compile with Apple's `xcrun metal` (the real check; runs in CI on macos-latest);
* macOS + build/kf_metal_run: run on the Metal GPU and compare with NumPy.
"""
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import pytest

from conftest import ROOT, assert_close
from kernelforge import zoo
from kernelforge.compiler import compile_graph
from kernelforge.graph import evaluate, random_feeds
from kernelforge.runtime import METAL_RUNNER, as_f32, run_msl
from kernelforge.target import M1_MSL

NAMES = sorted(zoo.SMALL)


def _cxx():
    for c in ("clang++", "g++"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        return None


def test_msl_shim_syntax_all():
    """One translation unit with every zoo program (one compiler start: zig c++ is slow to start)."""
    cxx = _cxx()
    if cxx is None:
        pytest.skip("no C++ compiler")
    parts = []
    for i, name in enumerate(NAMES):
        src = compile_graph(zoo.SMALL[name](), M1_MSL).source()
        src = src.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
        parts.append(f"namespace z{i} {{\n{src}\n}}")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "all.metal.cpp")
        with open(f, "w") as fh:
            fh.write("#include <metal_stdlib>\nusing namespace metal;\n" + "\n".join(parts))
        # a real -c compile (zig's driver mishandles -fsyntax-only); the object file is discarded
        r = subprocess.run(cxx + ["-std=c++17", "-O0", "-c", "-o", os.path.join(d, "all.o"), "-Wno-unknown-attributes",
                                  "-Wno-attributes", "-I", os.path.join(ROOT, "metal", "shim"), f],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]


def _has_xcrun_metal():
    if platform.system() != "Darwin":
        return False
    r = subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True)
    return r.returncode == 0


@pytest.mark.parametrize("name", NAMES)
def test_msl_xcrun_compile(name):
    if not _has_xcrun_metal():
        pytest.skip("Apple Metal compiler (xcrun metal) not available: macOS only")
    prog = compile_graph(zoo.SMALL[name](), M1_MSL)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "k.metal")
        with open(f, "w") as fh:
            fh.write(prog.source())
        r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", f, "-o", os.path.join(d, "k.air")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]


_probe_cache: list = []


def _metal_device():
    """Device name, or None when the machine has no Metal device (checked explicitly with
    `kf_metal_run --probe`, exit code 3). Any other probe failure is an error, not a skip."""
    if not _probe_cache:
        r = subprocess.run([str(METAL_RUNNER), "--probe"], capture_output=True, text=True, timeout=60)
        if r.returncode == 3:
            _probe_cache.append(None)
        else:
            assert r.returncode == 0, f"Metal probe failed: {r.stderr}"
            import json
            _probe_cache.append(json.loads(r.stdout)["device"])
    return _probe_cache[0]


@pytest.mark.parametrize("name", NAMES)
def test_msl_runs_on_metal_gpu(name):
    if not METAL_RUNNER.exists():
        pytest.skip("build/kf_metal_run not built (macOS only: see metal/kf_metal_run.m)")
    if _metal_device() is None:
        pytest.skip("no Metal device on this machine (kf_metal_run --probe)")
    g = zoo.SMALL[name]()
    f = random_feeds(g, 2)
    vals = evaluate(g, f)
    res = run_msl(compile_graph(g, M1_MSL), f)
    for o in g.outputs:
        assert_close(as_f32(res["data"][o.name], o.shape), vals[o.name], 2e-3, name)
