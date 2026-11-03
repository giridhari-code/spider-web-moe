# Full Spider-Web training pipeline

The executable order is **pretraining → mid-training (continued/domain-adaptive pretraining) → SFT → DPO**. Mid-training is a practical bridge stage; the original `TRAINING.md` specifies raw-corpus pretraining, SFT and optional DPO. All inputs are pre-tokenized because this repository does not yet ship a production tokenizer or chat-template implementation.

## Data contracts

- **Pretrain / midtrain:** one or more raw `uint32` token streams (currently one file per stage). Little-endian 4-byte IDs, no header. `seq_len + 1` consecutive tokens make input and next-token targets. Split train/validation before tokenization; never validate on overlapping text.
- **SFT:** JSONL, one record per line: `{"input_ids":[...],"labels":[...]}`. `labels` must be `-100` for system/user/prompt tokens and token IDs for assistant answer tokens. If omitted, all tokens are supervised. Rows are truncated/padded to `seq_len`; padding labels are `-100`.
- **DPO:** JSONL: `{"prompt_ids":[...],"chosen_ids":[...],"rejected_ids":[...]}`. Prompt is context; chosen/rejected are completions. The reference policy is loaded from `reference_checkpoint` (recommended: SFT best checkpoint) and frozen. DPO optimizes the standard logistic preference objective with beta.

A tokenizer must reserve consistent special token IDs and produce all stage files using the same vocabulary as `model.vocab_size`. Check that every ID is in `[0, vocab_size)`.

## Run

```bash
pip install -e .
# Edit paths and enable desired stages in configs/train-pipeline.yaml
python -m spider_moe.cli train-pipeline --config configs/train-pipeline.yaml --device auto
# Resume a stage checkpoint (stage selection is configured by resume_stage)
python -m spider_moe.cli train-pipeline --config configs/train-pipeline.yaml --resume checkpoints/pretrain/latest.pt
```

`enabled: false` skips a stage. Stage order is fixed. Start with a small model/short sequence and a few hundred steps; only scale after the overfit test, validation split and GPU memory checks pass. T4 uses FP16 automatically when BF16 is unavailable; RTX cards with BF16 support use BF16. Explicit `precision: fp16|bf16|auto` is supported.

## Operational behavior

- AdamW, linear warmup + cosine decay, gradient accumulation, global gradient clipping, AMP/GradScaler, periodic atomic checkpoints, best-validation checkpoint and resumable optimizer/scaler/RNG state.
- Pretrain/midtrain/SFT validation loss and CE/aux/edge logging. DPO logs preference loss and reward margin.
- SFT uses masked next-token CE; only labels other than `-100` contribute.
- DPO currently computes chosen/rejected sequence log-probabilities individually; this is correctness-oriented and not throughput-optimized. DPO's validation/reward-model evaluation and distributed training are not yet implemented.
- The model's recurrent loop runs full depth during training. Early exit is inference-only.
- `grad_accum` accumulates gradients across microbatches. Dataset sampling is randomized with a stage seed. Checkpoints preserve RNG state, but do not yet preserve the exact sampled-example cursor; a resumed run is reproducible at the RNG level, not guaranteed bit-for-bit identical across dataloader boundaries.

## Recommended staged workflow

1. **Pretrain:** broad, high-quality, deduplicated corpus. Verify train and held-out loss fall; save `checkpoints/pretrain/best.pt`.
2. **Midtrain:** domain-specific or recent high-quality corpus, lower LR and fewer tokens. Initialize from the same in-memory model after pretraining (the pipeline does this automatically).
3. **SFT:** instruction/chat examples with assistant-only labels. Keep validation conversations held out by source/document, not random line split.
4. **DPO:** preference pairs with prompt/chosen/rejected token IDs. Reference checkpoint should normally be the SFT best checkpoint. Keep preference data separate from eval.

## Limitations to address before production

No bundled tokenizer, raw-text preprocessing/deduplication, packed multi-document stream builder, multi-shard weighted sampler, distributed/FSDP training, activation-checkpoint toggle, LoRA, reward-model training, DPO eval, or production experiment tracker. The current implementation is a single-process PyTorch reference pipeline suitable for smoke tests and small-scale experiments, not a validated 100M–1B training system. The performance budgets in `TRAINING.md` are targets, not measured results for this code.

Optional `init_checkpoint` on any stage loads its model weights before that stage starts. This is useful when running only SFT or mid-training as a separate invocation. When running the complete pipeline, weights flow forward in memory automatically; do not set `init_checkpoint` unless intentionally overriding them.

## Tokenization helpers (optional Transformers install)

```bash
pip install -e '.[tokenizers]'
python scripts/tokenize_text.py --input corpus.txt --output data/tokenized/pretrain/train.bin --tokenizer YOUR_TOKENIZER --eos
python scripts/tokenize_sft.py --input sft_messages.jsonl --output data/sft/train.jsonl --tokenizer YOUR_TOKENIZER
python scripts/tokenize_preferences.py --input preference_text.jsonl --output data/preferences/train.jsonl --tokenizer YOUR_TOKENIZER
```

The SFT source format is `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`. Assistant-only masking requires a chat template with `{% generation %}` markers and a Transformers version that returns `assistant_masks`. Preference source format is `{"prompt":"...","chosen":"...","rejected":"..."}`; use a consistent prompt/chat serialization for all three fields. The helpers are convenience converters, not a replacement for corpus cleaning, licensing checks, deduplication, or split hygiene.
