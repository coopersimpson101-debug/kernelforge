"""Analytic cost model used to rank schedule candidates before measuring them.

predicted_time = max(DRAM time, FMA time, shared-memory time) / parallel efficiency + overheads

* DRAM bytes are estimated per template from the tiling (e.g. GEMM re-reads A once per column
  block and B once per row block; a stencil without staging pays halo re-reads through L1);
* memory-level parallelism follows Little's law: the bytes in flight (resident threads x bytes each
  thread requests per iteration) must cover bandwidth x latency, else bandwidth is scaled down;
* coalescing: a warp's request is charged in 32-byte sectors, so lanes that read 4 bytes 128 bytes
  apart pay 8x;
* fixed costs: kernel launch, barriers, reduction trees.

It is deliberately simple (a few dozen lines per template). How well it ranks real schedules is
measured, not assumed: ``bench/tune.py --full-sweep`` times every candidate of one space and
reports the Spearman rank correlation between predicted and measured times.
"""
from __future__ import annotations

import math

from .compute import ComputeDef
from .quant import BLOCK_BYTES, QK
from .target import Target

LATENCY_S = 600e-9  # DRAM round trip (order of magnitude for GDDR6 / LPDDR4X)
LAUNCH_S = 4e-6
BARRIER_S = 40e-9


def _wbytes_per_elem(dtype: str) -> float:
    if dtype in BLOCK_BYTES:
        return BLOCK_BYTES[dtype] / QK
    return 2.0 if dtype == "f16" else 4.0


def _mlp_eff(threads_total: int, bytes_per_thread: float, t: Target) -> float:
    resident = min(threads_total, t.cores * 1024)
    in_flight = resident * bytes_per_thread
    need = t.gbps * 1e9 * LATENCY_S
    return max(0.05, min(1.0, in_flight / need))


def _wave_eff(groups: int, t: Target, per_core: int) -> float:
    slots = t.cores * max(1, per_core)
    waves = math.ceil(groups / slots)
    return groups / (waves * slots)


def predict(tmpl: str, cd: ComputeDef, kn: dict, t: Target) -> float:
    bw = t.gbps * 1e9
    peak = t.gflops * 1e9
    out_bytes = 4 * math.prod(cd.out.shape)
    in_bytes = sum(math.prod(i.shape) * _wbytes_per_elem(i.dtype) for i in cd.inputs)

    if tmpl == "matvec":
        N, K = cd.meta["N"], cd.meta["K"]
        R, T, U, V = kn["R"], kn["T"], kn["U"], kn["V"]
        wb = _wbytes_per_elem(cd.meta["wdtype"])
        threads = R * T
        groups = math.ceil(N / R)
        blocked = cd.meta.get("form") == "blocked"
        per_thread = U * (QK * wb if blocked else V * wb)
        # coalescing: a warp covers 32/T rows, each row T consecutive chunks of per_thread bytes
        sector_eff = max(0.4, min(1.0, T * per_thread / 32.0))
        KE = K // QK if blocked else K
        sector_eff *= min(1.0, KE / (T * U * V))  # lanes left idle when a row is shorter than a sweep
        mlp = _mlp_eff(groups * threads, per_thread, t)
        occ = _wave_eff(groups, t, max(1, 1024 // threads))
        dram = (N * K * wb + 4 * K + 4 * N) / (bw * sector_eff)
        flops = 2.0 * N * K / (peak * 0.25)  # dequant + fma, ~4 ops/element
        red = 0.0 if T == 1 else (math.log2(T) * BARRIER_S if kn["xlane"] == "shared" else 20e-9)
        if blocked:
            # FITTED, not derived: on the RTX 2060 the per-thread fixed cost of the dequantising
            # block loop (~1 us per lane, measured over T = 2..32 for 4864x896 Q4_0) dominates when
            # many lanes share a short row; this one constant makes the model prefer few lanes/row
            red += N * T / (t.cores * 32) * 1.0e-6
        stage = (math.ceil(K / (T * U * V * (QK if blocked else 1))) * 2 * BARRIER_S) if kn.get("stage_x") else 0
        iters = math.ceil(K / (T * U * V * (QK if blocked else 1)))
        loop = iters * 30e-9 / max(1, U)
        return max(dram / (mlp * max(occ, 0.5)), flops) + red + stage + loop * (N / (t.cores * 32 * R) + 1) * 0.05 + LAUNCH_S

    if tmpl == "matmul":
        M, N, K = cd.meta["M"], cd.meta["N"], cd.meta["K"]
        BM, BN, BK, TM, TN = kn["BM"], kn["BN"], kn["BK"], kn["TM"], kn["TN"]
        wb = _wbytes_per_elem(cd.meta["wdtype"])
        a_bytes, b_bytes = 4 * M * K, wb * N * K
        threads = (BM // TM) * (BN // TN)
        groups = math.ceil(M / BM) * math.ceil(N / BN)
        if kn.get("stage", 1):
            traffic = a_bytes * math.ceil(N / BN) + b_bytes * math.ceil(M / BM)
            l2 = 0.5  # fraction of re-reads served by L2
            dram = (a_bytes + b_bytes + l2 * (traffic - a_bytes - b_bytes) * 0.3 + out_bytes) / bw
            smem_reads = M * N * K * ((1 / TN + 1 / TM) if kn.get("reg") else 2.0) * 4
            smem_bw = t.cores * 128 * 1.5e9
            smem = smem_reads / smem_bw
            barriers = math.ceil(K / BK) * 2 * BARRIER_S * math.ceil(groups / (t.cores * 2))
        else:
            dram = (a_bytes * N / max(1, TN) + b_bytes * M / max(1, TM)) * 0.05 / bw
            smem, barriers = 0.0, 0.0
        regs = TM * TN + TM + TN
        ilp = min(1.0, regs / 24.0) if regs <= 96 else 96.0 / regs
        occ = _wave_eff(groups, t, max(1, 1024 // threads) if regs <= 64 else 1)
        fma = 2.0 * M * N * K / (peak * max(0.15, ilp) * max(occ, 0.3))
        return max(dram, fma, smem) + barriers + LAUNCH_S

    if tmpl == "stencil":
        wx, wy, wz, cz = kn["wx"], kn["wy"], kn["wz"], kn["cz"]
        nodes = math.prod(cd.out.shape)
        threads = wx * wy * wz
        groups = math.ceil(nodes / (threads * cz))
        halo = (1 + 2.0 / wx + 2.0 / wy + 2.0 / (wz * cz))
        base = in_bytes + out_bytes
        if kn.get("stage"):
            dram = base * (1 + 0.25 * (halo - 1)) / bw
            extra = 2 * BARRIER_S * math.ceil(groups / (t.cores * max(1, 1024 // threads)))
        else:
            sector_eff = min(1.0, wx * 4 / 32.0)
            dram = base * (1 + 0.6 * (halo - 1)) / (bw * sector_eff)
            extra = 0.0
        mlp = _mlp_eff(groups * threads, 12.0 * cz, t)
        return dram / mlp + extra + LAUNCH_S

    if tmpl == "row":
        rows = math.prod(cd.out.shape[:-1]) or 1
        wg = kn["wg"]
        nred = sum(1 for _ in cd.reductions)
        passes = 1 if kn.get("stage") else max(1, nred + 1)
        dram = (in_bytes * min(passes, 2) + out_bytes) / bw
        red = nred * (math.log2(wg) if kn["xlane"] == "shared" else 2) * BARRIER_S
        occ = _wave_eff(rows, t, max(1, 1024 // wg))
        mlp = _mlp_eff(rows * wg, 8.0, t)
        return dram / (mlp * max(occ, 0.3)) + red + LAUNCH_S

    # injective
    wg, vec, ept = kn["wg"], kn["vec"], kn["ept"]
    n = math.prod(cd.out.shape)
    mlp = _mlp_eff(n // (vec * ept), 4.0 * vec * ept * max(1, len(cd.inputs)), t)
    return (in_bytes + out_bytes) / (bw * mlp) + LAUNCH_S
