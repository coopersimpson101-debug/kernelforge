// Hand-written baseline: WGSL port of tinyinfer's Metal `silu_mul` (g = silu(g) * u, in place).
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<uniform> n: i32;

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) i: vec3<u32>) {
  if (i32(i.x) < n) {
    let v = g[i.x];
    g[i.x] = v / (1.0 + exp(-v)) * u[i.x];
  }
}
