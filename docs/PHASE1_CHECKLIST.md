# Phase 1 complete checklist

## A. Documentation
- [x] `docs/PHASE1_NATIVE_ORBIT.md`
- [x] `docs/NATIVE_GRAPH_CONTRACT.md`
- [x] `docs/NATIVE_WEIGHT_FORMAT.md`
- [x] `docs/PHASE1_CHECKLIST.md`

## B. Kernel ABI
- [x] `kernels/include/spider_kernels.h` — f32 + f16 stubs + caps + moe_loop_step
- [x] `kernels/include/spider_cuda.h`
- [x] Existing scalar / ARM / CUDA retained
- [x] `kernels/src/phase1_api_ext.cpp` — f16 stubs + caps
- [x] Expanded `kernels/tests/test_kernels.cpp`

## C. Python bridge
- [x] `kernels/python/spider_native.py`

## D. Weights
- [x] `spider_infer/orbit/weight_loader.py`
- [x] `scripts/export_native_weights.py`

## E. Orbit
- [x] `spider_infer/orbit/native_backend.py`
- [ ] Wire `OrbitEngine.from_model(..., native_spiderw=path)` one-liner on main

## F. Known gaps (documented)
1. edge_value fusion (torch parity) — Phase 1.1
2. Expert CUDA grouped GEMM — Phase 1.5
3. FP16 compute ABI reserved (-2)
4. Prelude/coda attention still torch
5. Dynamic per-token CSR — static/exported in Phase 1
6. C moe_loop_step compose returns -3; Python compose is Phase-1 path

## Integrate
```bash
cmake -S . -B build -DSPIDER_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build --output-on-failure
python scripts/export_native_weights.py --checkpoint checkpoints/latest.pt --out model.spiderw --build-csr-placeholder
python -c "from spider_infer.orbit.native_backend import NativeBackend; print(NativeBackend.from_spiderw('model.spiderw').snapshot())"
```
