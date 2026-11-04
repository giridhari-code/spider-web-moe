/* Phase 1.5: tiled expert up+GELU then down, with workspace. */
#include "spider_kernels.h"
#include "spider_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) return (int)e; } while(0)

__device__ __forceinline__ float gelu_tanh(float a) {
    float c = 0.7978845608f * (a + 0.044715f * a * a * a);
    return 0.5f * a * (1.f + tanhf(c));
}

__global__ void expert_up_gelu_kernel(
    const float* __restrict__ x, int B, int H, int U, int E, int K,
    const int32_t* __restrict__ ids, const float* __restrict__ up,
    float* __restrict__ act) {
    int b = blockIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;
    int e = ids[b * K + k];
    extern __shared__ float sh_x[];
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        sh_x[i] = x[b * H + i];
    __syncthreads();
    if (e < 0 || e >= E) {
        for (int u = threadIdx.x; u < U; u += blockDim.x)
            act[(b * K + k) * U + u] = 0.f;
        return;
    }
    const float* W = up + (size_t)e * H * U;
    for (int u = threadIdx.x; u < U; u += blockDim.x) {
        float a = 0.f;
        for (int i = 0; i < H; ++i)
            a += sh_x[i] * W[(size_t)i * U + u];
        act[(b * K + k) * U + u] = gelu_tanh(a);
    }
}

__global__ void expert_down_kernel(
    const float* __restrict__ act, int B, int H, int U, int E, int K,
    const int32_t* __restrict__ ids, const float* __restrict__ rw,
    const float* __restrict__ down, float* __restrict__ out) {
    int b = blockIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;
    int e = ids[b * K + k];
    if (e < 0 || e >= E) return;
    float r = rw[b * K + k];
    const float* y = act + (size_t)(b * K + k) * U;
    const float* W = down + (size_t)e * U * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float s = 0.f;
        for (int u = 0; u < U; ++u)
            s += y[u] * W[(size_t)u * H + j];
        atomicAdd(&out[b * H + j], r * s);
    }
}

extern "C" size_t spider_cuda_expert_workspace_bytes(int B, int K, int U) {
    if (B <= 0 || K <= 0 || U <= 0) return 0;
    return (size_t)B * (size_t)K * (size_t)U * sizeof(float);
}

extern "C" int spider_cuda_expert_grouped_gemm_v2_f32(
    const float* x, int B, int H, int U, int E, int K,
    const int32_t* ids, const float* rw,
    const float* up, const float* down,
    float* out, float* workspace, void* stream) {
    if (!x || !ids || !rw || !up || !down || !out || !workspace ||
        B <= 0 || H <= 0 || U <= 0 || E <= 0 || K <= 0)
        return 1;
    cudaStream_t st = (cudaStream_t)stream;
    CUDA_OK(cudaMemsetAsync(out, 0, (size_t)B * H * sizeof(float), st));
    dim3 grid(B, K);
    int t_up = 128;
    if (H > 256) t_up = 256;
    if (H < 32) t_up = 32;
    int t_dn = 128;
    size_t shmem = (size_t)H * sizeof(float);
    expert_up_gelu_kernel<<<grid, t_up, shmem, st>>>(x, B, H, U, E, K, ids, up, workspace);
    expert_down_kernel<<<grid, t_dn, 0, st>>>(workspace, B, H, U, E, K, ids, rw, down, out);
    return (int)cudaGetLastError();
}
