/** Print the WebGPU adapter Dawn picks, its features and limits:  node harness/probe.ts */
import { openGpu } from './gpu.ts';

const gpu = await openGpu();
console.log(JSON.stringify({ adapter: gpu.adapter, features: gpu.features }, null, 2));
void gpu.keepAlive;
