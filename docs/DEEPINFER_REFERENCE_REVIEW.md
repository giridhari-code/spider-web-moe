# DeepInfer Reference Review — Spider-Web Optimization Baseline

## Scope

Reviewed the uploaded `DeepInfer-main.zip` as a source-code reference and compared its relevant inference/runtime patterns with the current `spider-web-moe` implementation. This is a static source review, not a benchmark or a claim that DeepInfer can be built on the current machine.

## 1. What DeepInfer contains

DeepInfer is a DeepSeek V4.1-Flash inference runtime. Its README lists:
- Model/runtime and sampler kernels.
- Attention DP / balanced CP and MoE expert parallelism with EPLB.
- DSpark / confidence-scheduled verification.
- SWA bounded replay.
- Asynchronous scheduling and batch-invariant inference.
- Structured output.

Its documented build target is Linux, Python 3.12, GCC >= 13, CUDA 13 for NVIDIA, or CANN 9.2 for Ascend. The README lists NVIDIA SM100 and Huawei Ascend 950 as supported hardware. This is not a generic CUDA/ARM backend.

### Main code areas

| Area | Main source files | Responsibility |
|---|---|---|
| Request engine | `deepinfer/engine/llm_engine.py` | Request lifecycle, scheduling orchestration, output handling |
| Native scheduler | `csrc/engine/scheduler.hpp` | Waiting/running queues, prefill/decode scheduling, chunking, preemption |
| Block allocator | `csrc/engine/block_manager.hpp` | Physical block IDs, free-list, reference counts, pin/free |
| Cache layout | `deepinfer/worker/cache_engine.py` | Cache byte layouts, per-layer views, state layout, GPU allocation |
| Attention | `deepinfer/model_executor/models/v4_modules/attn.py` | Prefill/decode attention paths, cache writes/loads, SWA/index/compressed KV |
| Cache kernels | `deepinfer/model_executor/layers/tilelang_ops/{store_kv_cache,load_paged_kv_cache}.py` | Paged cache store/load, quantized cache layouts |
| Kernel dispatch | `deepinfer/model_executor/layers/tilelang_ops/ops.py` | Selects CUDA or Ascend TileLang backend |
| CUDA graphs | `deepinfer/worker/cuda_graph_runner.py` | Captures/replays fixed decode graph shapes |
| Model structure | `deepinfer/model_executor/models/v4_modules/` | MLA attention, MoE, cross-layer sharing, YoCo, MTP |

## 2. Paged KV/cache findings

The cache implementation is more substantial than a simple tensor cache:

1. `CacheEngine` allocates a large GPU byte buffer indexed by physical blocks. Its block shape is derived from model attention configuration.
2. `GPUCacheInterface._kvcache_layout()` and `KVCacheManager.process()` carve per-layer cache views out of that shared allocation.
3. `KVCacheManager` handles model-specific compressed KV, sparse index KV, sliding-window attention (SWA), and cross-layer cache sharing.
4. `store_kv_cache()` writes token records using a `slot_mapping`; `load_paged_kv_cache()` loads compact source block ranges using segment metadata.
5. `csrc/engine/block_manager.hpp` maintains free physical block IDs and reference counts. Pinning supports shared/prefix blocks; freeing returns a block only when its reference count reaches zero.
6. `csrc/engine/scheduler.hpp` owns admission, prefill/decode scheduling, chunked prefill, decode block allocation, and preemption. It also handles forked sequences and copy-on-write of a partially filled final block.
7. The TileLang kernels support specific BF16 / FP8 / FP4 cache formats and scale layouts. They are specialized kernels, not drop-in generic attention.

**README nuance:** The README says a complete KV-cache system is required for a complete inference service and is not included in the release. However, the source does include lower-level cache allocation/layout, paged cache store/load kernels, and a physical block allocator. Treat the README statement as a warning that the release is not a complete deployable serving/cache product—not as evidence that there is no cache code in the repository.

## 3. What maps well to Spider-Web

Useful design patterns to adapt:
- One shared physical block pool / compact block IDs rather than allocating a full page pool independently for every layer and loop.
- Explicit logical-block-to-physical-block tables.
- `slot_mapping` for append/store and segment descriptors for gather/load.
- Reference-counted blocks for safe prefix sharing and copy-on-write.
- Separate cache *layout metadata* from physical storage allocation.
- Preallocate device-resident metadata buffers if CUDA graph replay will consume block tables.
- A scheduler that budgets both sequence slots and token/block capacity.
- Benchmark cache store/load, allocation, and attention separately.

These are architectural patterns to reimplement against Spider-Web's own tensor layout. DeepInfer's model-specific compressed cache format and SM100/Ascend kernels should not be copied blindly.

## 4. Important mismatches in the current Spider-Web code

### 4.1 Attention and loop dimensions

`spider_moe/model.py` currently has attention only in the prelude and coda `TransformerLayer`s. The shared recurrent `SpiderMoELayer` is router/graph/expert/normalization logic and contains no attention operation.

Therefore, the current model's live KV cache naturally belongs to the actual attention modules (prelude/coda), not automatically to every recurrent loop. The older `INFERENCE.md` describes a per-loop KV dimension, but the current executable model does not implement per-loop attention. These two descriptions need a deliberate architecture decision before wiring cache tensors into the model.

### 4.2 Current `PagedKVCache` limitations

`spider_moe/paged_kv.py` is a correctness-oriented allocator/reference:
- It allocates a separate full K/V page pool for every `(layer, loop)` pair.
- `num_pages` is per pool, not a global page budget.
- `gather()` materializes contiguous K/V tensors rather than letting attention read physical pages directly.
- `block_table()` constructs a fresh tensor on each call; it is not a stable preallocated device buffer for graph replay.
- There is no reference-counted sharing, copy-on-write, scheduler, eviction, or CUDA paged-attention kernel.
- It is not connected to `SpiderWebLM.generate()`.

For the current model, retaining the loop axis without adding loop-specific attention would allocate unnecessary cache pools.

### 4.3 Current model hot paths

`spider_moe/model.py` reveals higher-priority compute bottlenecks than low-level cache kernels:
- `SpiderMoELayer.forward()` materializes all `edge_value` projections for all experts before sparse destination selection.
- It loops over source experts, edge slots, and expert FFNs in Python.
- The expert FFN loop evaluates an expert over the full `[B,T,H]` message tensor and masks its contribution afterwards. This is not grouped sparse expert execution.
- The dynamic edge generator emits `[B,T,E,E]` logits, which grows quadratically with expert count.
- Early-exit masking preserves outputs for completed positions, but the recurrent layer is still called on the full batch/token tensor. It does not yet skip inactive-token computation.
- `SpiderWebLM.generate()` recomputes the complete prompt/context on every generated token. It has no incremental prefill/decode state and no KV-cache integration.

These should be profiled and addressed before expecting the current CUDA reference kernels to produce large end-to-end speedups.

## 5. Recommended optimization sequence

### Phase 0 — lock down semantics and baseline
- Decide whether recurrent loops contain attention. If not, cache only prelude/coda attention layers; if yes, implement those attention operations before adding a loop cache dimension.
- Add deterministic correctness tests for prefill vs incremental decode equivalence.
- Add benchmark cases for prefill, one-token decode, router, edge routing/message passing, expert FFN, and cache append/gather.
- Record peak memory, latency, and tokens/sec on the target device.

### Phase 1 — model-level incremental inference
- Refactor `CausalSelfAttention` to support prefill and one-token decode with `past_key_values`.
- Thread per-layer cache state through `TransformerLayer` and `SpiderWebLM`.
- Make `generate()` prefill once, then process only the newly generated token.
- Validate logits from cached decode against the current full-context implementation.

### Phase 2 — correct paged allocator design
- Replace per-(layer,loop) duplicate full pools with a global physical block budget and a clear page ownership/layout scheme.
- Store cache for each attention layer in a packed per-block layout (or a well-defined set of layer slices).
- Keep block tables and slot mappings in preallocated device tensors.
- Add reference counts and copy-on-write only after single-request correctness is stable.

### Phase 3 — sparse MoE execution
- Replace Python expert loops with grouped/indexed expert execution.
- Avoid computing all expert edge transforms when only a sparse subset is needed.
- Define and test exact graph aggregation semantics before fusing kernels.
- Implement real active-token compaction or active-index execution for early exit.

### Phase 4 — native kernel integration
- Profile first, then optimize the dominant operations.
- Add PyTorch/native parity tests for each kernel.
- Use backend-specific implementations only where hardware is available; retain a scalar correctness reference.

### Phase 5 — serving runtime
- Add request block budgeting, continuous batching, preemption, and page recycling.
- Add CUDA graph capture only after stable device-resident metadata and cache addresses exist.

## 6. Environment and validation constraints

The current execution environment reports:
- Linux x86_64.
- Python 3.13.
- PyTorch 2.10 CPU-only.
- No CUDA device visible.
- No verified CUDA toolkit / NVIDIA runtime.

Consequences:
- The uploaded DeepInfer project cannot be validated here against its documented Python 3.12 + CUDA 13 + SM100 or Ascend 950 target.
- CUDA and Ascend performance claims cannot be measured here.
- CPU-side logic, static inspection, reference tests, and scalar correctness tests can still be developed.

## 7. First concrete optimization recommendation

Start with **incremental prefill/decode attention**, not copying DeepInfer's compressed cache format or writing more CUDA kernels immediately.

Reason: `generate()` currently re-runs the entire context for each new token. A cache-aware attention path removes repeated prefix projection/attention work and provides the correctness foundation for paging later. The cache layout should initially match the actual prelude/coda attention layers. Once this is verified, evolve the allocator to a global block pool and connect block tables to a paged attention kernel.
