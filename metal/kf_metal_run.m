// kernelforge Metal harness: the macOS twin of harness/run.ts. Same JSON job format, but the
// pipeline code is Metal Shading Language and every step also carries "threads" (threadgroup size).
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation metal/kf_metal_run.m -o build/kf_metal_run
//   build/kf_metal_run job.json            (prints a JSON result to stdout)
//
// MSL is compiled at run time with newLibraryWithSource (no offline toolchain needed). Timing: each
// run is one command buffer with R repetitions of the steps; GPUEndTime - GPUStartTime per run / R.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static void fail(NSString *msg) {
  fprintf(stderr, "kf_metal_run: %s\n", msg.UTF8String);
  exit(1);
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc < 2) fail(@"usage: kf_metal_run <job.json>");
    NSString *jobPath = [NSString stringWithUTF8String:argv[1]];
    NSString *dir = [jobPath stringByDeletingLastPathComponent];
    NSData *jd = [NSData dataWithContentsOfFile:jobPath];
    if (!jd) fail(@"cannot read job");
    NSError *err = nil;
    NSDictionary *job = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&err];
    if (!job) fail(err.localizedDescription);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) fail(@"no Metal device");
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSMutableDictionary<NSString *, id<MTLBuffer>> *bufs = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *inits = [NSMutableDictionary dictionary];
    NSDictionary *bspec = job[@"buffers"];
    for (NSString *name in bspec) {
      NSUInteger bytes = [bspec[name][@"bytes"] unsignedIntegerValue];
      NSUInteger size = MAX(16, ((bytes + 4 + 15) / 16) * 16);
      bufs[name] = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
      NSString *init = bspec[name][@"init"];
      if (init && ![init isKindOfClass:[NSNull class]]) {
        inits[name] = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:init]];
      }
    }

    NSMutableArray *results = [NSMutableArray array];
    for (NSDictionary *c in job[@"cases"]) {
      NSMutableDictionary *res = [NSMutableDictionary dictionaryWithObject:c[@"id"] forKey:@"id"];
      @try {
        for (NSString *name in bufs) {  // reset
          id<MTLBuffer> b = bufs[name];
          memset(b.contents, 0, b.length);
          NSData *d = inits[name];
          if (d) memcpy(b.contents, d.bytes, MIN(d.length, b.length));
        }
        NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipes = [NSMutableDictionary dictionary];
        for (NSDictionary *p in c[@"pipelines"]) {
          id<MTLLibrary> lib = [dev newLibraryWithSource:p[@"code"] options:nil error:&err];
          if (!lib) @throw [NSException exceptionWithName:@"compile" reason:err.localizedDescription userInfo:nil];
          id<MTLFunction> fn = [lib newFunctionWithName:p[@"entry"]];
          if (!fn) @throw [NSException exceptionWithName:@"compile" reason:@"entry point not found" userInfo:nil];
          id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
          if (!ps) @throw [NSException exceptionWithName:@"pipeline" reason:err.localizedDescription userInfo:nil];
          pipes[p[@"name"]] = ps;
        }
        NSArray *steps = c[@"steps"];
        void (^encode)(id<MTLComputeCommandEncoder>, NSInteger) = ^(id<MTLComputeCommandEncoder> enc, NSInteger reps) {
          for (NSInteger r = 0; r < reps; r++) {
            for (NSDictionary *s in steps) {
              id<MTLComputePipelineState> ps = pipes[s[@"pipeline"]];
              [enc setComputePipelineState:ps];
              NSArray *bnd = s[@"bindings"];
              for (NSUInteger i = 0; i < bnd.count; i++) [enc setBuffer:bufs[bnd[i]] offset:0 atIndex:i];
              NSArray *g = s[@"groups"], *t = s[@"threads"];
              [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntegerValue], [g[1] unsignedIntegerValue], [g[2] unsignedIntegerValue])
                  threadsPerThreadgroup:MTLSizeMake([t[0] unsignedIntegerValue], [t[1] unsignedIntegerValue], [t[2] unsignedIntegerValue])];
              [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
          }
        };
        {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          encode(enc, 1);
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          if (cb.error) @throw [NSException exceptionWithName:@"exec" reason:cb.error.localizedDescription userInfo:nil];
          NSMutableDictionary *outs = [NSMutableDictionary dictionary];
          for (NSString *name in c[@"readback"]) {
            NSString *file = [NSString stringWithFormat:@"%@.%@.out.bin", c[@"id"], name];
            NSUInteger bytes = [bspec[name][@"bytes"] unsignedIntegerValue];
            NSData *d = [NSData dataWithBytes:bufs[name].contents length:bytes];
            [d writeToFile:[dir stringByAppendingPathComponent:file] atomically:NO];
            outs[name] = file;
          }
          res[@"outputs"] = outs;
        }
        NSInteger runs = [c[@"runs"] integerValue], R = MAX(1, [c[@"repeat"] integerValue]);
        NSInteger warm = c[@"warmup"] ? [c[@"warmup"] integerValue] : 2;
        if (runs > 0) {
          NSMutableArray *times = [NSMutableArray array];
          for (NSInteger it = 0; it < warm + runs; it++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, R);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / (double)R;
            if (it >= warm) [times addObject:@(ns)];
          }
          NSArray *sorted = [times sortedArrayUsingSelector:@selector(compare:)];
          res[@"times_ns"] = times;
          res[@"median_ns"] = sorted[sorted.count / 2];
          res[@"timer"] = @"MTLCommandBuffer.GPUStartTime/GPUEndTime";
          res[@"repeat"] = @(R);
        }
        res[@"ok"] = @YES;
      } @catch (NSException *e) {
        res[@"ok"] = @NO;
        res[@"error"] = e.reason ?: e.name;
      }
      [results addObject:res];
    }
    NSDictionary *out = @{@"adapter" : @{@"device" : dev.name}, @"features" : @[], @"errors" : @[], @"cases" : results};
    NSData *od = [NSJSONSerialization dataWithJSONObject:out options:0 error:&err];
    fwrite(od.bytes, 1, od.length, stdout);
  }
  return 0;
}
