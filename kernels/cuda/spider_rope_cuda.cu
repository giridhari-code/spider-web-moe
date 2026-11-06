/* Phase 2.4.1: CUDA RoPE precompute + apply. */
#include "spider_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void rope_precompute_kernel(
    int head_dim, int max_T, float theta, float* cos_out, float* sin_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_T) return;
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = 1.f / powf(theta, (2.f * i) / (float)head_dim);
        float ang = (float)t * freq;
        cos_out[t * half + i] = cosf(ang);
        sin_out[t * half + i] = sinf(ang);
    }
}

__global__ void rope_apply_kernel(
    float* x, int B, int n_heads, int T, int head_dim,
    const float* cos_table, const float* sin_table, int position) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_heads * T;
    if (idx >= total) return;
    int half = head_dim / 2;
    int t = idx % T;
    int tmp = idx / T;
    int h = tmp % n_heads;
    int b = tmp / n_heads;
    int abs_t = position + t;
    float* row = x + ((((size_t)b * n_heads + h) * T + t) * head_dim);
    const float* c = cos_table + (size_t)abs_t * half;
    const float* s = sin_table + (size_t)abs_t * half;
    for (int i = 0; i < half; ++i) {
        float x0 = row[i];
        float x1 = row[i + half];
        row[i] = x0 * c[i] - x1 * s[i];
        row[i + half] = x0 * s[i] + x1 * c[i];
    }
}

extern "C" int spider_cuda_rope_precompute_f32(
    int head_dim, int max_T, float theta, float* cos_out, float* sin_out, void* stream) {
    if (!cos_out || !sin_out || head_dim <= 0 || max_T <= 0 || head_dim % 2 != 0 || theta <= 0.f)
        return 1;
    cudaStream_t st = (cudaStream_t)stream;
    int thr = 128;
    int blocks = (max_T + thr - 1) / thr;
    rope_precompute_kernel<<<blocks, thr, 0, st>>>(head_dim, max_T, theta, cos_out, sin_out);
    return (int)cudaGetLastError();
}

extern "C" int spider_cuda_rope_apply_f32(
    float* x, int B, int n_heads, int T, int head_dim,
    const float* cos_table, const float* sin_table, int position, void* stream) {
    if (!x || !cos_table || !sin_table || B <= 0 || n_heads <= 0 || T <= 0 ||
        head_dim <= 0 || head_dim % 2 != 0 || position < 0)
        return 1;
    cudaStream_t st = (cudaStream_t)stream;
    int total = B * n_heads * T;
    int thr = 128;
    int blocks = (total + thr - 1) / thr;
    rope_apply_kernel<<<blocks, thr, 0, st>>>(
        x, B, n_heads, T, head_dim, cos_table, sin_table, position);
    return (int)cudaGetLastError();
}
