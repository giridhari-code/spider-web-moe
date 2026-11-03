# Global paged KV cache and paged attention — Phase 2

## Storage design

The previous Phase 1 allocator created an independent full K/V page pool per `(layer, loop)`. Phase 2 replaces that design for model inference with one global physical page pool.

K and V storage:
`[physical_pages, attention_layers, page_size, num_heads, head_dim]`

A physical page therefore stores one token block for every attention layer. Each request has one logical block table shared across those layers. `num_pages` is the global physical page budget.

This matches the current executable model: attention exists in prelude and coda Transformer layers. The recurrent SpiderMoE layer currently has no attention, so it does not get a cache dimension.

## Allocator

`spider_moe/global_paged_kv.py` implements:
- A single global free-page list and request-slot pool.
- Stable device-resident `block_tables[max_requests, max_blocks]`.
- Stable device-resident `sequence_lengths[max_requests]`.
- Physical-page reference counts.
- Prefix sharing via `fork_prefix()`.
- Copy-on-write when a request appends into a shared partial page.
- Page recycling on request free.

The fixed block-table row for a request is updated in-place; the row's data pointer remains stable. This is useful groundwork for CUDA Graph replay. The Python allocator still performs host-side allocation and table updates, so it is not yet a graph-captured serving allocator.

## Attention backends

`spider_moe/paged_attention.py` provides:
1. PyTorch reference: reads logical pages and materializes K/V for SDPA. This is the correctness path and supports CPU.
2. Optional Triton CUDA single-token decode kernel: reads the block table and K/V pages directly and computes online softmax attention without concatenating K/V pages.

Prefill and multi-token decode chunks currently use the PyTorch reference path. The Triton kernel is a prototype, not a tuned FlashAttention replacement. It has not been compiled or run in this CPU-only development environment. Runtime compilation errors fall back to the reference path.

## Model API

- `model.allocate_global_paged_cache(...)`
- `model.prefill_paged(input_ids, cache, request_ids)`
- `model.decode_step_paged(input_ids, cache, request_ids, early_exit=True)`
- `model.generate_paged(input_ids, cache, request_ids, ...)`

The batch decode API currently requires equal sequence lengths across requests. Continuous batching with heterogeneous lengths needs a per-row query position/length path and a scheduler.

## Memory planning

Approximate cache bytes:
`2 * num_pages * num_attention_layers * page_size * num_heads * head_dim * bytes_per_element`

For FP16/BF16, bytes_per_element is 2. This is the global page budget. The page pool is preallocated, so choose `num_pages` based on free VRAM after model weights and activations.

## Tests

`tests/test_global_paged_kv.py` covers shared global allocation, stable block-table views, prefix sharing/COW, free/recycle, and allocation rollback.

`tests/test_paged_model.py` compares paged prefill and decode logits against full-context recomputation, and checks batched request tables and generation.

## Remaining work

- Verify Triton kernel compilation/execution on T4 SM75 and RTX devices.
- Add paged-prefill and multi-token chunk kernels.
- Add slot-mapping batched append kernel.
- Add asynchronous allocator and scheduler integration.
- Add CUDA Graph capture only after device-side block table updates and cache addresses are stable.
- Add heterogeneous-length batch decode and continuous batching.
