// Hand-written baseline: a line-by-line WGSL port of tinyinfer's Metal decode kernel `matvec_q8_0`
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

// struct block_q8_0 { half d; char qs[32]; } = 34 bytes
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> a: MatvecArgs;

// little-endian 4 bytes at byte offset o (any alignment); 2 bytes as half at even o
fn ld32(o: i32) -> u32 {
  let s = u32(o & 3) * 8u;
  let w0 = w[o >> 2];
  if (s == 0u) { return w0; }
  return (w0 >> s) | (w[(o >> 2) + 1] << (32u - s));
}
fn ldhalf(o: i32) -> f32 { return unpack2x16float(w[o >> 2] >> (u32(o & 2) * 8u)).x; }

@compute @workgroup_size(256)
fn matvec_q8_0(@builtin(workgroup_id) tg: vec3<u32>, @builtin(local_invocation_index) li: u32,
               @builtin(subgroup_invocation_id) lane: u32) {
  let sg = i32(li) / 32;
  let r = i32(tg.x) * ROWS_PER_TG + sg;
  if (r >= a.rows) { return; }
  let nb = a.cols / QK;
  let wr = r * nb;
  var sum = 0.0;
  for (var b = i32(lane); b < nb; b += 32) {
    let blk = (wr + b) * 34;
    let xb = b * (QK / 4);
    var s = 0.0;
    for (var j = 0; j < QK / 4; j++) {
      let q = vec4<f32>(unpack4xI8(ld32(blk + 2 + 4 * j)));
      s += dot(q, x[xb + j]);
    }
    sum += s * ldhalf(blk);
  }
  sum = subgroupAdd(sum);
  if (lane == 0u) { y[r] = sum; }
}
