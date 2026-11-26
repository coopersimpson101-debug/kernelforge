// Verbatim excerpt of tinyinfer/metal/kernels.metal (MIT, same author): the decode matvec
// kernels and silu_mul, used unchanged as the hand-written Metal baseline on Apple GPUs.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define ROWS_PER_TG 8

struct block_q8_0 { half d; char qs[QK]; };        // 34 bytes, same as ggml
struct block_q4_0 { half d; uchar qs[QK / 2]; }; // 18 bytes, same as ggml

struct MatvecArgs { int rows; int cols; };

kernel void matvec_f32(device const float *w [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant MatvecArgs &a [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    device const float4 *wr = (device const float4 *)(w + (ulong)r * a.cols);
    device const float4 *x4 = (device const float4 *)x;
    float sum = 0.0f;
    for (int i = int(lane); i < a.cols / 4; i += 32) sum += dot(wr[i], x4[i]);
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_f16(device const half *w [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant MatvecArgs &a [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    device const half4 *wr = (device const half4 *)(w + (ulong)r * a.cols);
    device const float4 *x4 = (device const float4 *)x;
    float sum = 0.0f;
    for (int i = int(lane); i < a.cols / 4; i += 32) sum += dot(float4(wr[i]), x4[i]);
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_q8_0(device const block_q8_0 *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant MatvecArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    const int nb = a.cols / QK;
    device const block_q8_0 *wr = w + (ulong)r * nb;
    float sum = 0.0f;
    for (int b = int(lane); b < nb; b += 32) {
        device const float4 *xb = (device const float4 *)(x + b * QK);
        float s = 0.0f;
        for (int j = 0; j < QK / 4; j++) {
            float4 q = float4(wr[b].qs[4 * j], wr[b].qs[4 * j + 1], wr[b].qs[4 * j + 2], wr[b].qs[4 * j + 3]);
            s += dot(q, xb[j]);
        }
        sum += s * float(wr[b].d);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void matvec_q4_0(device const block_q4_0 *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant MatvecArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg) * ROWS_PER_TG + int(sg);
    if (r >= a.rows) return;
    const int nb = a.cols / QK;
    device const block_q4_0 *wr = w + (ulong)r * nb;
    float sum = 0.0f;
    for (int b = int(lane); b < nb; b += 32) {
        device const float *xb = x + b * QK;
        float s = 0.0f;
        for (int j = 0; j < QK / 2; j++) {
            const uchar v = wr[b].qs[j];
            s += float(int(v & 0x0F) - 8) * xb[j] + float(int(v >> 4) - 8) * xb[j + QK / 2];
        }
        sum += s * float(wr[b].d);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[r] = sum;
}

kernel void silu_mul(device float *g [[buffer(0)]],
                     device const float *u [[buffer(1)]],
                     constant int &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (int(i) < n) {
        const float v = g[i];
        g[i] = v / (1.0f + exp(-v)) * u[i];
    }
}

