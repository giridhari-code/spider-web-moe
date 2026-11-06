/* Phase 2.4: CUDA causal attention + KV cache (FP32). */
#include "spider_kernels.h"
#include "spider_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) return (int)e; } while(0)

__global__ void attn_qkv_proj_kernel(
    const float* __restrict__ x, const float* __restrict__ W_qkv,
    float* __restrict__ qkv, int B, int T, int H) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b >= B || t >= T) return;
    const float* xin = x + (size_t)(b * T + t) * H;
    float* row = qkv + (size_t)(b * T + t) * (3 * H);
    for (int o = threadIdx.x; o < 3 * H; o += blockDim.x) {
        const float* w = W_qkv + (size_t)o * H;
        float acc = 0.f;
        for (int i = 0; i < H; ++i) acc += xin[i] * w[i];
        row[o] = acc;
    }
}

__global__ void attn_write_kv_kernel(
    const float* __restrict__ qkv, float* __restrict__ keys, float* __restrict__ values,
    int B, int T, int H, int n_heads, int max_T, int position) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b >= B || t >= T) return;
    int D = H / n_heads;
    int abs_t = position + t;
    const float* row = qkv + (size_t)(b * T + t) * (3 * H);
    for (int h = threadIdx.x; h < n_heads; h += blockDim.x) {
        for (int d = 0; d < D; ++d) {
            size_t koff = ((((size_t)b * n_heads + h) * max_T + abs_t) * D + d);
            keys[koff] = row[H + h * D + d];
            values[koff] = row[2 * H + h * D + d];
        }
    }
}

__global__ void attn_scores_softmax_v_kernel(
    const float* __restrict__ qkv, const float* __restrict__ keys,
    const float* __restrict__ values, float* __restrict__ ctx,
    int B, int T, int H, int n_heads, int max_T, int position) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int ti = blockIdx.z;
    if (b >= B || h >= n_heads || ti >= T) return;
    int D = H / n_heads;
    int abs_q = position + ti;
    float scale = rsqrtf((float)D);
    const float* q = qkv + (size_t)(b * T + ti) * (3 * H) + h * D;

    float m = -INFINITY;
    __shared__ float sm[256];
    for (int kj = threadIdx.x; kj <= abs_q; kj += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < D; ++d) {
            size_t koff = ((((size_t)b * n_heads + h) * max_T + kj) * D + d);
            dot += q[d] * keys[koff];
        }
        m = fmaxf(m, dot * scale);
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    m = sm[0];

    float sum = 0.f;
    for (int kj = threadIdx.x; kj <= abs_q; kj += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < D; ++d) {
            size_t koff = ((((size_t)b * n_heads + h) * max_T + kj) * D + d);
            dot += q[d] * keys[koff];
        }
        sum += expf(dot * scale - m);
    }
    sm[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.f / fmaxf(sm[0], 1e-20f);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int kj = 0; kj <= abs_q; ++kj) {
            float dot = 0.f;
            for (int dd = 0; dd < D; ++dd) {
                size_t koff = ((((size_t)b * n_heads + h) * max_T + kj) * D + dd);
                dot += q[dd] * keys[koff];
            }
            float a = expf(dot * scale - m) * inv;
            size_t voff = ((((size_t)b * n_heads + h) * max_T + kj) * D + d);
            acc += a * values[voff];
        }
        ctx[((size_t)b * T + ti) * H + h * D + d] = acc;
    }
}

__global__ void attn_out_proj_kernel(
    const float* __restrict__ ctx, const float* __restrict__ W_o,
    float* __restrict__ out, int B, int T, int H) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b >= B || t >= T) return;
    const float* cin = ctx + (size_t)(b * T + t) * H;
    float* orow = out + (size_t)(b * T + t) * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float* w = W_o + (size_t)j * H;
        float acc = 0.f;
        for (int i = 0; i < H; ++i) acc += cin[i] * w[i];
        orow[j] = acc;
    }
}

extern "C" int spider_cuda_attention_causal_kv_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    float* keys, float* values, int max_T, int position, int* length_inout,
    float* out, float* workspace, void* stream) {
    if (!x || !W_qkv || !W_o || !keys || !values || !out || !workspace ||
        B <= 0 || T <= 0 || H <= 0 || n_heads <= 0 || H % n_heads != 0 || position < 0)
        return 1;
    if (position + T > max_T) return 3;
    cudaStream_t st = (cudaStream_t)stream;

    float* qkv = workspace;
    float* ctx = workspace + (size_t)B * T * 3 * H;

    dim3 grid_bt(B, T);
    int thr = 128;
    attn_qkv_proj_kernel<<<grid_bt, thr, 0, st>>>(x, W_qkv, qkv, B, T, H);
    attn_write_kv_kernel<<<grid_bt, thr, 0, st>>>(qkv, keys, values, B, T, H, n_heads, max_T, position);

    dim3 grid_s(B, n_heads, T);
    attn_scores_softmax_v_kernel<<<grid_s, 128, 0, st>>>(
        qkv, keys, values, ctx, B, T, H, n_heads, max_T, position);

    attn_out_proj_kernel<<<grid_bt, thr, 0, st>>>(ctx, W_o, out, B, T, H);

    if (length_inout) {
        int nl = position + T;
        if (nl > *length_inout) *length_inout = nl;
    }
    return (int)cudaGetLastError();
}

extern "C" size_t spider_cuda_attention_kv_workspace_bytes(int B, int T, int H) {
    if (B <= 0 || T <= 0 || H <= 0) return 0;
    return (size_t)B * (size_t)T * (size_t)(3 * H + H) * sizeof(float);
}
