// Hand-written baseline: a line-by-line WGSL port of tinyinfer's Metal decode kernel `matvec_f32`
// (tinyinfer/metal/kernels.metal). tinyinfer has this kernel only in Metal (GPU) and C (CPU), so it
// was ported to WGSL by hand to time it on the RTX 2060, keeping the same work decomposition:
//   * one SIMD-group (32 lanes) per output row, ROWS_PER_TG = 8 rows per threadgroup (256 threads);
//   * lanes stride over float4 chunks (f32) or 32-element blocks (Q8_0 / Q4_0);
//   * simd_sum -> subgroupAdd (needs subgroup size 32: true on NVIDIA and Apple GPUs);
//   * MatvecArgs {rows, cols} is a uniform, exactly like Metal's `constant MatvecArgs &`.
// Deviation forced by WGSL: no 8-bit storage type, so the 34/18-byte ggml blocks are read from an
// array<u32> with (unaligned) word loads + shifts. Metal's early `return` before simd_sum is kept;
// WGSL's subgroup-uniformity diagnostic is switched off for it (the row index is uniform per SIMD-group).
enable subgroups;
diagnostic(off, subgroup_uniformity);

struct MatvecArgs { rows: i32, cols: i32 };
const QK: i32 = 32;
const ROWS_PER_TG: i32 = 8;

@group(0) @binding(0) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> a: MatvecArgs;

@compute @workgroup_size(256)
fn matvec_f32(@builtin(workgroup_id) tg: vec3<u32>, @builtin(local_invocation_index) li: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
  let sg = i32(li) / 32;
  let r = i32(tg.x) * ROWS_PER_TG + sg;
  if (r >= a.rows) { return; }
  let c4 = a.cols / 4;
  let wr = r * c4;
  var sum = 0.0;
  for (var i = i32(lane); i < c4; i += 32) { sum += dot(w[wr + i], x[i]); }
  sum = subgroupAdd(sum);
  if (lane == 0u) { y[r] = sum; }
}
