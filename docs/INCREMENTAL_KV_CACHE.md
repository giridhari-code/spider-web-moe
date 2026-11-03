# Incremental attention and KV cache — Phase 1

## Design decision from the executable model

`SpiderWebLM` has attention in the prelude and coda `TransformerLayer`s. The shared recurrent `SpiderMoELayer` is router + sparse graph + expert FFNs + normalization and does not currently contain attention. Thus Phase 1 allocates K/V only for actual attention layers.

This is intentionally different from the older inference document's conceptual `[layer, loop, ...]` cache. A loop axis is only valid if attention is implemented inside each recurrent loop.

## Runtime API

- `model.allocate_kv_cache(batch_size, max_seq_len=None, dtype=None, device=None)`
- `model.prefill(input_ids, cache=None) -> (logits, cache)`
- `model.decode_step(input_ids, cache, early_exit=True) -> (logits, cache)`
- `model.generate(...)` now uses prefill once and then incremental decode.

The cache preallocates `[batch, heads, max_seq_len, head_dim]` K and V tensors for each attention layer. K/V for each new token are written into the cache in-place. Attention reads the valid prefix only.

For decode chunks, an offset causal mask ensures query position `p+i` can attend to keys up to `p+i`. The standard prefill path uses PyTorch SDPA's causal mode.

When the cache reaches capacity, `generate()` re-prefills the most recent configured context window. The cache capacity is sized to `min(config.max_seq_len, prompt_len + max_new_tokens)` when possible.

## Correctness checks

`tests/test_kv_decode.py` validates:
- Prefill logits against the original full forward.
- Token-by-token cached decode logits against full-context recomputation.
- Chunked decode offset-causal masking.
- Capacity checks and sliding-window re-prefill in generation.

## Known limitations

- Contiguous static K/V buffers, not paged attention.
- The model uses learned absolute position embeddings. Full forward, prefill, incremental decode, and paged decode all use the same absolute position IDs; cache position semantics are covered by parity tests.
- Cache uses the model parameter dtype by default. For best memory use, convert model weights to FP16 on T4/RTX before allocating cache. T4 is an FP16 target; do not assume BF16 acceleration.
- No prefix sharing, block allocator integration, CUDA graph capture, or fused attention kernel yet.
- Decode early exit compacts active batch rows for the recurrent loop. It does not yet compact individual tokens inside a multi-token chunk.

## Hardware support strategy

The implementation uses `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to select an available backend. On a Colab T4 or RTX GPU, benchmark FP16 and FP32 separately. The exact SDPA kernel depends on GPU architecture, PyTorch build, dtype, head dimension, and mask shape. No GPU performance claim is made until run on the target hardware.

## Benchmarking on Colab T4 / RTX

Run `python benchmarks/benchmark_incremental_decode.py --device cuda --dtype fp16`. The script checks first-step logit parity, warms both paths, synchronizes CUDA around timing, and reports milliseconds per generated token and speedup. Run `--dtype fp32` as a comparison. Use the same prompt length, batch size, model config, and GPU when comparing results. Do not compare CPU timings to GPU timings.
