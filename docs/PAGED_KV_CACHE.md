# Paged KV cache — implementation notes

## Source design followed

The implementation follows `INFERENCE.md` §3:
- Page size defaults to 16 tokens.
- Logical blocks map to physical pages through a block table.
- The cache is loop-aware and keeps independent pools per `(layer, loop)`.
- Request completion recycles physical pages.
- Block tables can be copied/exposed as device-resident tensors for indirect addressing.

## API

- `PagedKVCache.allocate(request_id, seq_len, layer, loop)` reserves pages.
- `append(request_id, layer, loop, keys, values)` appends `[T, heads, head_dim]` K/V.
- `gather(request_id, layer, loop)` returns contiguous K/V in token order.
- `block_table(request_id, layer, loop)` returns device `int32` logical-to-physical mapping.
- `free(request_id)` releases all pages owned by the request.
- `stats()` reports aggregate page utilization.

Each `(layer, loop)` has its own page pool, as recommended in §3.5. The cache stores K and V separately, with shape `[physical_pages, page_size, heads, head_dim]`.

## Deliberate limits in this milestone

- This is a PyTorch allocator/reference, not a CUDA paged-attention kernel.
- `gather()` copies the request's pages into contiguous tensors. A fused attention kernel should consume the block table directly instead.
- No prefix sharing/refcounts, eviction, scheduler, chunked prefill, or continuous batching yet.
- The existing `SpiderWebLM` attention layers are not yet loop-specific and do not accept cache state. Consequently this class is not automatically invoked by `generate()`.
- The cache allocates one independent page pool per layer-loop pair. Its `num_pages` argument is per pool, not a global count.

## Smoke tests

`tests/test_paged_kv.py` covers page-boundary append/gather, independent loop/layer pools, recycling, exhaustion rollback, and device block-table formatting.
