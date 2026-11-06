# Phase 1 — Native Orbit (PyTorch-free inference path)

**Goal:** Serve Spider/Sentinel through Orbit without calling PyTorch ops on the
hot path. Training may stay in PyTorch; inference loads native weights and runs
C/CUDA/ARM kernels.

## Stack naming

| Name | Role |
|------|------|
| Sentinel | Product / model family |
| Spider | Architecture (looped MoE + attention shells) |
| Orbit | Serving runtime (continuous batching, early-exit, CUDA graphs) |
| Dobu | Expert weight-chunk residency (CPU↔GPU LRU) |
| Forge | Kernel fabric (attention / GEMM / EP / select / tile) |
| Native kernels | `kernels/` C ABI + CUDA + NEON |

## Deliverables (nothing optional in Phase 1 scope)

1. **Graph contract** — `docs/NATIVE_GRAPH_CONTRACT.md`
2. **Weight format** — `docs/NATIVE_WEIGHT_FORMAT.md` + exporter + loader
3. **Kernel API** — FP32 + FP16 headers, scalar/CUDA/ARM
4. **CUDA upgrades** — warp-friendly router notes, expert tile path hooks
5. **Parity tests** — expanded scalar smoke + CSR golden vectors
6. **Python bridge** — `kernels/python/spider_native.py` (ctypes, no torch required for call)
7. **Orbit native backend** — `spider_infer/orbit/native_backend.py`
8. **Orbit weight loader** — `spider_infer/orbit/weight_loader.py`
9. **Export script** — `scripts/export_native_weights.py`
10. **CMake** — CUDA optional, tests, shared libs
11. **Wire checklist** — how OrbitEngine switches to native decode loop

## Execution order

```
[1] Graph CSR contract locked
[2] Parity tests green (scalar)
[3] CUDA build + scalar↔CUDA parity on small tensors
[4] Export checkpoint → .spiderw
[5] Orbit native backend loads .spiderw and runs one loop step via C ABI
[6] FP16 path (storage + compute where supported)
[7] Fixed-loop CUDA graph capture (early_exit=off)
[8] Drop torch from Orbit decode path (model.forward no longer called)
```

## Explicit non-goals (Phase 1)

- Full training without PyTorch
- Tensor-core CUTLASS production GEMM (Phase 1.5 — interface reserved)
- Per-loop attention KV (SpiderMoE still has no attention)
- Complete continuous-batching rewrite in C++ (Python Orbit scheduler stays; compute goes native)

## Success criteria

- `ctest` / `spider_kernel_smoke` passes
- `python -c "from kernels.python.spider_native import NativeKernels; ..."` runs without importing torch
- `export_native_weights.py` produces loadable `.spiderw`
- `OrbitEngine` can be constructed with `native_compute=True` and complete one generate step
  without `torch.nn.Module.forward` on the recurrent block (attention shells may still use torch
  until Forge attention is fully native)

## Build

```bash
cmake -S . -B build -DSPIDER_BUILD_TESTS=ON
cmake --build build -j && ctest --test-dir build --output-on-failure

cmake -S . -B build-cuda -DSPIDER_ENABLE_CUDA=ON -DSPIDER_BUILD_TESTS=ON
cmake --build build-cuda -j
```
