// Extracted verbatim from flipgpu src/gpu/shaders/flip.wgsl by scripts/extract_flipgpu.py
// (pressure solve only: Jacobi + red-black SOR). Hand-written baseline; do not edit.
struct Params {
  dims: vec4<u32>,     // nx, ny, nz, numParticles
  p0: vec4<f32>,       // h, 1/h, dt, flipRatio
  gravity: vec4<f32>,  // body acceleration xyz (gravity + tank inertial force)
  mousePos: vec4<f32>, // xyz, radius
  mouseVel: vec4<f32>, // xyz, strength in [0,1] (0 = off)
  p1: vec4<f32>,       // kinematic viscosity nu, SOR omega, rest density, density-correction gain
  p2: vec4<f32>,       // push-out radius, unused, max particle speed, unused
};
const FLUID: u32 = 0u;
const AIR: u32 = 1u;
const SOLID: u32 = 2u;
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(6) var<storage, read_write> cellType: array<u32>;
@group(0) @binding(7) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(9) var<storage, read_write> rhs: array<f32>;
fn NX() -> u32 { return P.dims.x + 1u; }
fn NY() -> u32 { return P.dims.y + 1u; }
fn numNodes() -> u32 { return (P.dims.x + 1u) * (P.dims.y + 1u) * (P.dims.z + 1u); }
fn node(i: u32, j: u32, k: u32) -> u32 { return i + NX() * (j + NY() * k); }
fn nodeI(c: vec3<i32>) -> u32 { return node(u32(c.x), u32(c.y), u32(c.z)); }
fn coordOf(idx: u32) -> vec3<u32> {
  let nx = NX(); let ny = NY();
  return vec3<u32>(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}
fn inCells(c: vec3<i32>) -> bool {
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(P.dims.xyz));
}
fn typeAt(c: vec3<i32>) -> u32 {
  if (!inCells(c)) { return SOLID; }
  return cellType[nodeI(c)];
}
fn neighbourSum(c: vec3<i32>) -> vec2<f32> {
  var diag = 0.0;
  var s = 0.0;
  for (var e = 0u; e < 3u; e++) {
    for (var sgn = -1; sgn <= 1; sgn += 2) {
      var d = vec3<i32>(0);
      d[e] = sgn;
      let n = c + d;
      let t = typeAt(n);
      if (t != SOLID) { diag += 1.0; }
      if (t == FLUID) { s += pressure[nodeI(n)]; }
    }
  }
  return vec2<f32>(diag, s);
}
@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= numNodes()) { return; }
  let c = vec3<i32>(coordOf(idx));
  if (typeAt(c) != FLUID) { pressureOut[idx] = 0.0; return; }
  let ns = neighbourSum(c);
  pressureOut[idx] = select(0.0, (rhs[idx] + ns.y) / ns.x, ns.x > 0.0);
}
fn rbUpdate(idx: u32, parity: u32) {
  if (idx >= numNodes()) { return; }
  let q = coordOf(idx);
  if (((q.x + q.y + q.z) & 1u) != parity) { return; }
  let c = vec3<i32>(q);
  if (typeAt(c) != FLUID) { return; }
  let ns = neighbourSum(c);
  if (ns.x == 0.0) { return; }
  let gs = (rhs[idx] + ns.y) / ns.x;
  let w = P.p1.y;
  pressure[idx] = (1.0 - w) * pressure[idx] + w * gs;
}
@compute @workgroup_size(64)
fn rbRed(@builtin(global_invocation_id) id: vec3<u32>) { rbUpdate(id.x, 0u); }
@compute @workgroup_size(64)
fn rbBlack(@builtin(global_invocation_id) id: vec3<u32>) { rbUpdate(id.x, 1u); }
