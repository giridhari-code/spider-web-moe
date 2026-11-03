# Milestone 2: Wikipedia + custom BPE pretraining

## Pipeline

1. Stream the `wikimedia/wikipedia` dataset config `20231101.en`.
2. Normalize each page to one line and split by a stable BLAKE2 hash of page ID/title (default 2% validation).
3. Train byte-level BPE on training text only; reserve `<|endoftext|>`.
4. Encode train and validation text independently into little-endian uint32 streams with EOS between pages.
5. Set `model.vocab_size` from the actual tokenizer vocabulary.
6. Train only the pretraining stage using held-out validation windows and T4-safe starter settings.
7. Decode samples with the same tokenizer JSON.

## Colab commands

```bash
pip install -e ".[wikipedia]"
bash scripts/colab_milestone2.sh
```

## Expected artifacts

- `data/wikipedia/train.txt`, `validation.txt`
- `data/tokenizer/tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`
- `data/wikipedia_tokens/train.bin`, `validation.bin`
- `checkpoints/wikipedia-pretrain/latest.pt`, `best.pt`, periodic `step-*.pt`

## Guardrails

- Do not load the old synthetic-data checkpoints into this run.
- The tokenizer vocabulary size must exactly match the model config/checkpoint.
- Validation pages are separated before tokenizer training; validation text is never used to fit BPE merges.
- The token streams are uint32 little-endian, as required by `TokenBinaryDataset`.
- The current trainer samples random contiguous windows from one token stream. It is a simple single-corpus reference sampler, not a packed multi-shard production loader.
- Training and validation loss are useful signals but not sufficient to claim language quality. Evaluate decoded held-out samples and track repetition/diversity.

## Milestone 2 audit corrections

The Colab runner now trains byte-level BPE on a deterministic, evenly spaced sample of at most 10,000,000 training-text characters by default (`TOKENIZER_MAX_CHARS` overrides this; `0` uses the full training file). Validation pages are never used to fit tokenizer merges. The starter YAML is preserved; the actual vocabulary size is written to `configs/t4-wikipedia-pretrain.runtime.yaml`.

The language model now adds learned absolute position embeddings to token embeddings. The same absolute positions are used by full forward, contiguous KV prefill/decode, and paged KV prefill/decode. This is a deliberate model change from the earlier documented no-position-embedding reference; all pre-position-embedding checkpoints are incompatible and must not be resumed.

Validation output reports total objective, cross-entropy, and CE-derived perplexity separately. Perplexity is computed from CE, not from the auxiliary routing objective. These metrics remain small-sample diagnostics until evaluated over a larger held-out corpus.
