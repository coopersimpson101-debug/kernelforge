/**
 * Headless WebGPU device via Google's Dawn (the `webgpu` npm package), as in splatweb's GPU tests.
 * Requests timestamp queries, subgroups and the adapter's real compute limits when available.
 */
export interface Gpu {
  device: GPUDevice;
  features: string[];
  adapter: Record<string, unknown>;
  keepAlive: unknown[];
}

export async function openGpu(): Promise<Gpu> {
  const webgpu = await import('webgpu');
  Object.assign(globalThis, webgpu.globals);
  // timestamp_quantization rounds GPU timestamps for privacy in browsers; we want raw values.
  const gpu = webgpu.create(['disable-dawn-features=timestamp_quantization']);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const want: GPUFeatureName[] = ['timestamp-query', 'subgroups', 'shader-f16'];
  const requiredFeatures = want.filter((f) => adapter.features.has(f));
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: L.maxComputeWorkgroupSizeZ,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    },
  });
  const info = adapter.info as unknown as Record<string, unknown>;
  const desc: Record<string, unknown> = {};
  for (const k of ['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize']) {
    if (info && info[k] !== undefined) desc[k] = info[k];
  }
  desc.maxComputeInvocationsPerWorkgroup = L.maxComputeInvocationsPerWorkgroup;
  desc.maxComputeWorkgroupStorageSize = L.maxComputeWorkgroupStorageSize;
  return { device, features: requiredFeatures, adapter: desc, keepAlive: [gpu, adapter] };
}
