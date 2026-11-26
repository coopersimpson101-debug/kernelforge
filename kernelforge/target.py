"""Code-generation targets and the hardware numbers the cost model uses.

Hardware figures are vendor specifications (not measurements): RTX 2060 = TU106, 30 SMs, 1920
FP32 lanes at 1.68 GHz boost (6.45 TFLOP/s), 336 GB/s GDDR6; Apple M1 (8-core GPU) = 1024 lanes at
1.278 GHz (2.6 TFLOP/s), 68.25 GB/s LPDDR4X shared with the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    name: str  # wgsl | msl | c
    device: str = "generic"
    subgroup_size: int = 0  # 0 = no subgroup ops
    max_threads: int = 256
    max_shared: int = 16384  # bytes per workgroup
    max_groups: int = 65535
    cores: int = 30  # SMs / GPU cores
    gbps: float = 336.0
    gflops: float = 6450.0
    features: frozenset = field(default_factory=frozenset)

    @property
    def gpu(self) -> bool:
        return self.name != "c"


# WebGPU's portable defaults are 256 invocations / 16 KiB; the RTX 2060 adapter under Dawn reports
# 1024 / 32 KiB, which we request explicitly in the harness.
RTX2060_WGSL = Target("wgsl", "rtx2060", subgroup_size=32, max_threads=1024, max_shared=32768,
                      cores=30, gbps=336.0, gflops=6451.0, features=frozenset({"subgroups"}))
WGSL_PORTABLE = Target("wgsl", "portable", subgroup_size=0, max_threads=256, max_shared=16384)
M1_MSL = Target("msl", "m1", subgroup_size=32, max_threads=1024, max_shared=32768, cores=8,
                gbps=68.25, gflops=2617.0, features=frozenset({"subgroups"}))
C_REF = Target("c", "cpu", max_threads=1)

TARGETS = {"rtx2060": RTX2060_WGSL, "wgsl": WGSL_PORTABLE, "m1": M1_MSL, "c": C_REF}
