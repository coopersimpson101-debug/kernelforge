/**
 * kernelforge WebGPU harness: runs WGSL compute kernels described by a JSON job file, writes the
 * read-back buffers next to it and prints a JSON result to stdout.
 *
 *   node harness/run.ts <job.json>
 *
 * Job: { buffers: {name: {bytes, init?}}, cases: [{id, pipelines: [{name, code, entry}],
 *        steps: [{pipeline, bindings: [bufferName...] | {binding: bufferName}, groups: [x,y,z]}],
 *        readback: [names],
 *        repeat?: R, runs?: N, warmup?: W }] }
 * Timing: each run is ONE compute pass containing R repetitions of the case's steps, bracketed by
 * GPU timestamp queries (when the adapter supports them, else host wall-clock around
 * onSubmittedWorkDone). Runs are interleaved round-robin across cases. Reported per repetition in
 * nanoseconds, with the median.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { openGpu } from './gpu.ts';

interface BufSpec { bytes: number; init?: string }
interface Pipe { name: string; code: string; entry: string }
interface Step { pipeline: string; bindings: string[] | Record<string, string>; groups: [number, number, number] }
interface Case {
  id: string; pipelines: Pipe[]; steps: Step[]; readback?: string[];
  repeat?: number; runs?: number; warmup?: number; reset?: boolean;
}
interface Job { buffers: Record<string, BufSpec>; cases: Case[] }

const jobPath = process.argv[2];
if (!jobPath) {
  console.error('usage: node harness/run.ts <job.json>');
  process.exit(2);
}
const dir = dirname(jobPath);
const job = JSON.parse(readFileSync(jobPath, 'utf8')) as Job;

const gpu = await openGpu();
const { device } = gpu;
const errors: string[] = [];
device.addEventListener('uncapturederror', (e) => {
  errors.push((e as GPUUncapturedErrorEvent).error.message);
});

const bufs = new Map<string, GPUBuffer>();
const inits = new Map<string, Uint8Array>();
for (const [name, spec] of Object.entries(job.buffers)) {
  const size = Math.max(16, Math.ceil((spec.bytes + 4) / 16) * 16); // +4: unaligned word reads at the end
  bufs.set(name, device.createBuffer({
    label: name, size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  }));
  const data = new Uint8Array(size);
  if (spec.init) data.set(new Uint8Array(readFileSync(join(dir, spec.init))));
  inits.set(name, data);
}

function resetBuffers() {
  for (const [name, data] of inits) device.queue.writeBuffer(bufs.get(name)!, 0, data);
}

async function readBack(name: string): Promise<Uint8Array> {
  const src = bufs.get(name)!;
  const dst = device.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, src.size);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(dst.getMappedRange().slice(0));
  dst.unmap();
  dst.destroy();
  return out.subarray(0, job.buffers[name].bytes);
}

const hasTs = gpu.features.includes('timestamp-query');
const results: Record<string, unknown>[] = [];
type Prepared = { c: Case; res: Record<string, unknown>; encode: (p: GPUComputePassEncoder, reps: number) => void; times: number[] };
const prepared: Prepared[] = [];

// phase 1: compile + correctness run of every case (buffers reset to their initial contents first)
for (const c of job.cases) {
  const res: Record<string, unknown> = { id: c.id };
  results.push(res);
  try {
    const pipes = new Map<string, GPUComputePipeline>();
    for (const p of c.pipelines) {
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code: p.code, label: p.name });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length) {
        throw new Error(`WGSL compile error in ${p.name}: ${errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
      }
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: p.entry } });
      const err = await device.popErrorScope();
      if (err) throw new Error(`pipeline ${p.name}: ${err.message}`);
      pipes.set(p.name, pipeline);
    }
    const groups = c.steps.map((s) => {
      const pl = pipes.get(s.pipeline)!;
      return device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        // bindings: array (binding = index) or {binding: bufferName} for sparse layouts
        entries: (Array.isArray(s.bindings) ? s.bindings.map((b, i) => [i, b] as const)
          : Object.entries(s.bindings).map(([k, b]) => [Number(k), b] as const))
          .map(([binding, b]) => {
            const buffer = bufs.get(b);
            if (!buffer) throw new Error(`unknown buffer ${b}`);
            return { binding, resource: { buffer } };
          }),
      });
    });
    const encode = (pass: GPUComputePassEncoder, reps: number) => {
      for (let r = 0; r < reps; r++) {
        c.steps.forEach((s, i) => {
          pass.setPipeline(pipes.get(s.pipeline)!);
          pass.setBindGroup(0, groups[i]);
          pass.dispatchWorkgroups(s.groups[0], s.groups[1] ?? 1, s.groups[2] ?? 1);
        });
      }
    };
    if (c.reset !== false) resetBuffers();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encode(pass, 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const outs: Record<string, string> = {};
    for (const name of c.readback ?? []) {
      const file = `${c.id}.${name}.out.bin`;
      writeFileSync(join(dir, file), await readBack(name));
      outs[name] = file;
    }
    res.outputs = outs;
    res.ok = true;
    if ((c.runs ?? 0) > 0) prepared.push({ c, res, encode, times: [] });
  } catch (e) {
    res.ok = false;
    res.error = (e as Error).message;
  }
}

// phase 2: timing, interleaved round-robin across cases so that any other load on the GPU hits
// every candidate alike (each round times every case once)
if (prepared.length) {
  const qs = hasTs ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const resolve = hasTs ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const read = hasTs ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const rounds = Math.max(...prepared.map((p) => (p.c.warmup ?? 2) + (p.c.runs ?? 0)));
  for (let it = 0; it < rounds; it++) {
    for (const p of prepared) {
      const warm = p.c.warmup ?? 2;
      if (it >= warm + (p.c.runs ?? 0)) continue;
      const R = Math.max(1, p.c.repeat ?? 1);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      p.encode(pass, R);
      pass.end();
      if (qs) {
        enc.resolveQuerySet(qs, 0, 2, resolve!, 0);
        enc.copyBufferToBuffer(resolve!, 0, read!, 0, 16);
      }
      const t0 = process.hrtime.bigint();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      let ns = Number(process.hrtime.bigint() - t0);
      if (qs) {
        await read!.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(read!.getMappedRange().slice(0));
        read!.unmap();
        ns = Number(t[1] - t[0]);
      }
      if (it >= warm) p.times.push(ns / R);
    }
  }
  qs?.destroy();
  for (const p of prepared) {
    const sorted = [...p.times].sort((a, b) => a - b);
    p.res.times_ns = p.times;
    p.res.median_ns = sorted[Math.floor(sorted.length / 2)];
    p.res.timer = qs ? 'timestamp-query' : 'wall-clock';
    p.res.repeat = Math.max(1, p.c.repeat ?? 1);
  }
}

await device.queue.onSubmittedWorkDone();
void gpu.keepAlive;
process.stdout.write(JSON.stringify({ adapter: gpu.adapter, features: gpu.features, errors, cases: results }));
