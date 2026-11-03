# Native CUDA / ARM kernel implementation — phase 1

## Build

Scalar reference / x86:
```bash
cmake -S . -B build -DSPIDER_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

ARM64 (NEON):
```bash
cmake -S . -B build-arm -DSPIDER_BUILD_TESTS=ON
cmake --build build-arm -j
ctest --test-dir build-arm --output-on-failure
```

CUDA:
```bash
cmake -S . -B build-cuda -DSPIDER_ENABLE_CUDA=ON
cmake --build build-cuda -j
```
CUDA Toolkit and a compatible NVIDIA driver are required. Set `CMAKE_CUDA_ARCHITECTURES` to the actual target GPU when building for deployment.

## Kernel inventory

| Kernel | Scalar | ARM NEON | CUDA |
|---|---:|---:|---:|
| Router / top-k | reference | scalar fallback | CUDA kernel |
| Sparse CSR message passing | reference | scalar fallback | CUDA kernel |
| Expert FFN | reference | scalar fallback | CUDA correctness kernel |
| Residual + LayerNorm | reference | NEON vector affine pass | CUDA block kernel |
| Early-exit gate | reference | NEON dot product | CUDA block kernel |

The public host API in `spider_kernels.h` selects scalar or ARM at build time. CUDA entry points are separately named in `spider_cuda.h` and accept an opaque CUDA stream. Device buffers must already reside on the GPU.

## Current scope / honesty about optimization

This is the first native backend milestone, not the final tuned kernel suite. All native entry points currently accept FP32 contiguous data. ARM NEON accelerates the epilogue vector pass and exit dot product; graph and expert operations currently use scalar fallback. CUDA implementations are correctness-oriented initial kernels: the router and exit reductions are serial per row, graph uses one thread per hidden element, and the expert FFN uses a direct reference mapping rather than tensor-core grouped GEMM. They are useful for backend bring-up and parity work, not yet performance claims.

The kernel design notes specify FP32 accumulation, CSR sparse graph traversal, top-k routing, grouped expert GEMM, Welford normalization, and per-token early-exit. These implementations preserve those formulas at FP32, with the exception that the graph reference follows the supplied CSR pseudocode (weighted selected-source hidden accumulation). The higher-level PyTorch model currently has a different token-wise edge tensor interpretation; wire-up requires reconciling these representations before claiming end-to-end numerical parity.

## Next optimization steps

1. Add FP16/BF16 conversion and accumulation paths.
2. Replace CUDA serial row loops with warp reductions and block top-k.
3. Implement expert token permutation + grouped GEMM using cuBLASLt/CUTLASS or grouped tensor-core tiles.
4. Add NEON/SVE graph and expert microkernels, with thread-pool batch parallelism.
5. Add randomized scalar-vs-ARM-vs-CUDA parity tests for every kernel, then benchmarks on named hardware.
6. Add a PyTorch custom extension and dispatch from `SpiderMoELayer` only after parity passes.
