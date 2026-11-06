# Phase 1.5 — Expert grouped GEMM (CUDA + scalar path)

## Goal

Speed up sparse expert FFN after graph message passing:

```text
for token b, expert e=assign[b,k] weight r:
  a = messages[b] @ W_up[e]     # [H]x[H,U]
  g = GELU(a)
  delta[b] += r * (g @ W_down[e])
```

Weights: `W_up [E,H,U]`, `W_down [E,U,H]`.

## CUDA v2

1. **Up+GELU** — grid `(B,K)`, shared token `x[H]`, tile over `U` → workspace `act[B,K,U]`
2. **Down** — grid `(B,K)`, tile over `H`, `atomicAdd` into `out[b,:]`

```c
size_t spider_cuda_expert_workspace_bytes(int B, int K, int U);

int spider_cuda_expert_grouped_gemm_v2_f32(
    const float* tokens, int B, int H, int U, int E, int K,
    const int32_t* assign, const float* route,
    const float* W_up, const float* W_down,
    float* output, float* workspace, void* stream);
```

Legacy `spider_cuda_expert_grouped_gemm_f32` remains (v1, no workspace).

## Validation

Scalar vs hand reference max abs error < 1e-4 (`test_phase1_5`).
On GPU: v1 vs v2 same tolerance.

## Files

- `kernels/cuda/spider_expert_v2.cu`
- `kernels/include/spider_cuda.h`
- `kernels/tests/test_phase1_5.cpp`
- CMake: include `spider_expert_v2.cu` when `SPIDER_ENABLE_CUDA=ON`

## Next

CUTLASS/cuBLASLt when dense enough; FP16 expert weights; fuse GELU.
