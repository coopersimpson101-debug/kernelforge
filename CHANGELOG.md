# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-26

### Fixed
- Generated MSL returned NaN for `mlp` on a real Metal GPU: Metal's fast-math `tanh` behaves like
  (e^2x - 1)/(e^2x + 1) and overflows to inf/inf for |x| > ~44, which the fused GELU reaches. Both GPU
  backends now emit `tanh(clamp(x, -10, 10))` (tanh(10) is already 1.0 in float32). Regression graph
  `gelu_wide` (|x| up to ~200) runs on every backend.

### Changed
- The Metal GPU step in CI is blocking; it skips only when `kf_metal_run --probe` reports no Metal
  device. 18/18 zoo programs match NumPy on the macos-latest runner's GPU.

## [0.1.0] - 2026-11-26

### Added
- Tensor graph IR (elementwise, broadcast, reductions, matmul/linear, Q8_0/Q4_0/f16 weights,
  flipgpu-style Jacobi and red-black SOR stencils) with an independent NumPy interpreter.
- Operator fusion (injective chains, matmul epilogues, row reductions such as RMSNorm/softmax,
  stencil prologue/epilogue).
- TVM-style compute/schedule split; schedule templates for injective, row, matvec, GEMM and 3-D
  stencil kernels; bound-inference-based shared/register staging, unrolling, vectorisation,
  byte-load combining, bounds-check elimination, CSE.
- WGSL, Metal Shading Language and C backends; WebGPU (Dawn, Node) and Metal harnesses.
- Analytic cost model + grid-search autotuner measured on the GPU.
- Benchmarks vs hand-written tinyinfer matvec kernels (WGSL port) and flipgpu's pressure solver.
