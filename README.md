# Spider-Web Looped MoE — runnable reference code

This repository is a **PyTorch reference implementation** of the architecture in the supplied design documents: Transformer Prelude → shared recurrent Spider-MoE block → Transformer Coda → tied language-model head.

It includes model code, sparse top-k routing and graph message passing, training loss terms, binary token dataset loading, checkpointing, generation, configs, CLI, and tests.

## Important implementation scope

This is a runnable research/reference baseline, **not** a finished 400M-parameter production system. It now includes incremental contiguous KV caching and a global paged KV allocator with a PyTorch reference attention path plus an optional Triton single-token CUDA decode kernel. Production Rust orchestration, continuous batching, CUDA Graph capture, optimized prefill kernels, and a production tokenizer/server are still incomplete. The included CLI's byte-to-token mapping is only a smoke-test fallback; replace it with the tokenizer used to prepare your corpus.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e .
```

PyTorch installation may need a platform-specific command from https://pytorch.org/get-started/locally/.

## Quick smoke test

```bash
python - <<'PY'
import torch
from spider_moe import ModelConfig, SpiderWebLM
cfg = ModelConfig(vocab_size=512, hidden_size=64, num_heads=4,
                  num_experts=4, top_k=2, edges_per_expert=2,
                  num_loops=2, expert_ffn_size=128)
model = SpiderWebLM(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(x, targets=x, return_stats=True)
print(out["logits"].shape, out["loss"].item())
out["loss"].backward()
PY
```

## Prepare training data

The training loader expects a flat binary file of little-endian `uint32` token IDs, as specified in `TRAINING.md`:

```python
import numpy as np
np.asarray(token_ids, dtype="<u4").tofile("data/train.bin")
```

Create `data/` first. Use the exact same tokenizer/vocabulary as `model.vocab_size`.

## Train

For a small local validation run:

```bash
spider train --config configs/small-spider.yaml
```

For the design's larger default shape:

```bash
spider train --config configs/train-pretrain.yaml
```

Training uses next-token cross entropy plus weighted router/edge balance terms, full loop unrolling (no early exit), AdamW, gradient clipping, warmup/cosine decay, and checkpoint saving.

## Inference

```bash
spider infer --checkpoint checkpoints/latest.pt --prompt "Hello" --max-new-tokens 40
```

The fallback prompt encoder prints token IDs, not decoded natural language. Integrate a real tokenizer and decoder before using this as a chat model.

## Benchmark

```bash
spider bench --checkpoint checkpoints/latest.pt --batch-size 1 --seq-len 128
```

## Loop-aware Paged KV cache (phase 1)

`spider_moe.PagedKVCache` implements page allocation, append, gather, block tables, and page recycling. It uses page size 16 by default and maintains separate physical page pools for every `(layer, loop)` pair.

```python
import torch
from spider_moe import PagedKVCache

cache = PagedKVCache(
    num_layers=8, num_loops=6, num_pages=1024,
    page_size=16, num_heads=8, head_dim=64,
    dtype=torch.float16, device="cuda",
)
# K and V each have shape [tokens, heads, head_dim].
cache.append("request-1", layer=0, loop=0, keys=k, values=v)
k_all, v_all = cache.gather("request-1", layer=0, loop=0)
device_block_table = cache.block_table("request-1", layer=0, loop=0)
cache.free("request-1")
```

The block table is a device-resident `int32` tensor, padded with `-1`, suitable as the basis for indirect page addressing in a future CUDA graph attention kernel. `gather()` currently materializes contiguous K/V tensors for correctness and compatibility; it is not a zero-copy paged-attention kernel.

**Integration boundary:** The current reference model does not have loop-specific attention modules or a KV-cache-aware attention call. The cache is therefore implemented and tested as a runtime component, but not yet wired into `SpiderWebLM.generate()`. That integration must wait until the attention architecture and the source spec's per-loop attention parameters are reconciled.

## Incremental attention and KV cache (phase 1)

Generation now performs one prompt prefill and then sends only newly generated token(s) through `decode_step()`. K/V tensors are preallocated per actual attention layer (the prelude and coda Transformer layers) and updated in-place.

```python
logits, cache = model.prefill(prompt_ids)             # full prompt once
next_logits = logits[:, -1]
logits, cache = model.decode_step(next_token, cache)  # one token only
```

`generate()` uses this path automatically. When the cache reaches its context capacity, it rebuilds from the current sliding context. Cache capacity is sized to the expected generation length where possible, rather than always reserving the full configured context.

**Architecture note:** the current `SpiderMoELayer` has no attention operation. Therefore this implementation caches only prelude/coda attention layers; it does not create artificial per-loop KV tensors. The older paged allocator remains a separate reference component and is not yet wired to attention. The current cache is contiguous and static-capacity, not paged attention.

## Global paged KV cache + paged attention (phase 2)

Phase 2 adds `GlobalPagedKVCache`: one physical block pool shared by all requests and attention layers. Each physical block stores K/V for every actual attention layer, so the block table is shared across layers and `num_pages` is a **global** budget.

```python
cache = model.allocate_global_paged_cache(
    num_pages=512, page_size=16, max_requests=16,
    dtype=torch.float16, device="cuda",
)
logits, cache = model.prefill_paged(prompt_ids, cache, request_ids=["req-1"])
logits, cache = model.decode_step_paged(
    next_token_ids, cache, request_ids=["req-1"]
)
```

Features:
- Stable, preallocated device-resident block-table and sequence-length tensors.
- Shared physical page allocator across all requests and attention layers.
- Request slots and page recycling.
- Reference-counted prefix sharing with copy-on-write for a partially filled shared page.
- `prefill_paged()`, `decode_step_paged()`, and `generate_paged()` model APIs.
- Portable PyTorch paged-attention reference path.
- Optional Triton single-token CUDA decode kernel that reads K/V directly from physical pages without materializing a contiguous K/V sequence. Prefill and multi-token chunks currently use the PyTorch reference path.

The Triton kernel is optional. Install/enable a Triton build compatible with your PyTorch/CUDA environment to use it. The T4 (SM75) and RTX paths must be benchmarked on their actual devices; this development environment has no CUDA GPU, so CUDA kernel execution has not been verified here.

The prior `PagedKVCache` remains available for compatibility, but new work should use `GlobalPagedKVCache`. As before, cache pools cover only prelude/coda attention because the current recurrent `SpiderMoELayer` does not contain attention.

## Native CUDA / ARM kernels (phase 1)

The repository now includes a native C++ scalar reference, an ARM NEON backend, and initial CUDA kernels. See `docs/KERNEL_BACKENDS.md` for exact coverage and limitations.

```bash
# CPU / scalar (also used as parity reference)
cmake -S . -B build -DSPIDER_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# NVIDIA CUDA (requires CUDA Toolkit)
cmake -S . -B build-cuda -DSPIDER_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

ARM64 builds automatically select the NEON backend when CMake detects an ARM processor. The first native milestone is FP32 only. CUDA grouped expert GEMM and ARM graph/expert SIMD paths remain optimization work; these kernels are not yet wired into the Python model.

## Tests

```bash
pip install pytest
pytest -q
```

## Architecture mapping

| Design specification | Code |
|---|---|
| Prelude and Coda Transformer layers | `TransformerLayer` |
| Per-loop router bias and top-k | `SpiderMoELayer.router`, `loop_bias` |
| Loop embedding and dynamic sparse adjacency | `loop_embedding`, `edge_generator` |
| Sparse graph aggregation | selected `edge_indices` gather in `SpiderMoELayer.forward` |
| Expert FFNs and weighted combine | `experts` |
| Stability epilogue | `LayerNorm(h + αΔ + βx₀)` |
| Early-exit confidence head | `exit_head` |
| CE + auxiliary losses | `trainer.py` |

## Known research limitations

- The edge-balance regularizer is a simple uniform-target proxy; validate/tune it against the precise objective before large-scale training.
- Current graph aggregation is a clear tensor reference, not a fused sparse kernel. It is intentionally not advertised as fast.
- Early exit is implemented as a reference mask over sequence positions; it does not yet provide optimized per-request scheduler compaction.
- The attention stack is causal and uses PyTorch SDPA. Incremental and global paged KV paths are implemented for prelude/coda attention. The recurrent SpiderMoE layer has no attention today, so no per-loop KV pages are allocated. The optional Triton paged decode kernel still requires validation on target GPUs.
- Parameter count depends strongly on vocabulary, width, and expert FFN dimensions; the default 512-wide config is not guaranteed to equal the document's ~400M target.

See `docs/` for the original specifications.

## Full staged training

A staged runner now supports **pretraining → continued/domain mid-training → SFT → DPO**. Configure data paths and enable the desired stages in `configs/train-pipeline.yaml`, then run:

```bash
python -m spider_moe.cli train-pipeline --config configs/train-pipeline.yaml --device auto
```

See [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md) for the exact pre-tokenized uint32 and JSONL schemas, stage behavior, checkpoint/resume details, and limitations. Optional Transformers-based conversion helpers live in `scripts/` (`tokenize_text.py`, `tokenize_sft.py`, `tokenize_preferences.py`); install `.[tokenizers]` to use them. They still require a compatible tokenizer/chat template.

## Training stability note (embedding initialization)

The token embedding is tied to the language-model output projection. It is therefore
initialized with `ModelConfig.embedding_init_std` (default `0.02`) rather than
PyTorch's default `nn.Embedding` initialization (standard deviation near `1.0`).
This keeps initial tied-output logits in a sensible range and avoids an initially
saturated softmax. Checkpoints created before this fix retain their old embedding
weights when loaded; **start a fresh pretraining run** to benefit from the corrected
initialization. Do not resume the old collapsed checkpoint as the starting point.

The trainer also validates supervised labels, rejects batches containing only
ignored labels, and averages gradient-accumulation groups correctly when the final
group is shorter or a run is resumed. Non-finite losses now stop training instead
of silently skipping a batch and potentially applying a partial accumulated update.

## Milestone 2 — custom tokenizer + Wikipedia pretraining (Colab T4)

This workflow creates a deterministic held-out validation split from the English Wikipedia `20231101.en` snapshot using Hugging Face Datasets streaming, trains a byte-level BPE tokenizer only on training text, encodes both splits to little-endian `uint32`, synchronizes the model vocabulary size with the trained tokenizer, and launches the pretraining stage.

Install optional dependencies:

```bash
pip install -e ".[wikipedia]"
```

From the repository root (including in Colab):

```bash
bash scripts/colab_milestone2.sh
```

Defaults are 100,000 streamed pages, 2% held-out validation pages, an 8,192-token custom BPE vocabulary, and a small T4-oriented model (width 128, 4 experts, 2 recurrent loops, context 256, batch 2 × accumulation 8). These are starter settings, not a claim that this model is a high-quality language model. Change `WIKI_MAX_PAGES` and `VOCAB_SIZE` environment variables before running to adjust corpus/tokenizer scale.

The scripts can also be run individually:

```bash
python scripts/download_wikipedia.py --max-pages 100000
python scripts/train_custom_tokenizer.py --vocab-size 8192
python scripts/tokenize_wikipedia.py
python -m spider_moe.cli train-pipeline --config configs/t4-wikipedia-pretrain.yaml --device auto
```

After training, decode text with the exact tokenizer used for training:

```bash
python scripts/generate_wikipedia_model.py --checkpoint checkpoints/wikipedia-pretrain/best.pt --prompt "The history of science"
```

Important: verify the Wikipedia dataset revision, license/terms, and the downloaded text before large runs. Streaming page count is a capped sample, not the full encyclopedia. The deterministic page-hash split prevents the same page from entering both splits; it is not a complete near-duplicate removal pipeline. Keep the tokenizer JSON and its vocabulary/merges files with every checkpoint. The script updates `model.vocab_size` to the actual tokenizer size before training. The pretraining CLI consumes the tokenized binary files; it does not silently use the old byte-token fallback.

## Wikipedia pretraining audit update

The Milestone 2 Colab runner fits the custom byte-level BPE tokenizer on a deterministic sample (10M characters by default) rather than all downloaded text, keeps the starter YAML immutable, and writes a separate runtime config with the actual tokenizer vocabulary size. The model includes learned absolute position embeddings, applied consistently in full-sequence and KV-cache inference paths. This changes the checkpoint schema: checkpoints created before position embeddings were added cannot be loaded into this revision. Validation logs now distinguish total loss, cross-entropy, and CE-derived perplexity.

## Generation quality audit

Cached incremental generation now uses the same full recurrent-loop depth as training by default. Early exit is opt-in because the exit head is not trained by the language-model objective. See `docs/GENERATION_QUALITY_AUDIT.md`.
