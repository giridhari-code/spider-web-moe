# Full CodeBeast Audit — Milestone 2

## Scope

Reviewed the Milestone 2 Wikipedia downloader, tokenizer training, tokenization, binary dataset, model forward and KV-cache paths, training/evaluation/checkpoint logic, generation script, configuration handling, and test coverage.

## Corrections applied

1. **Bounded tokenizer training input.** The default pipeline previously passed the entire downloaded training text (the user's run produced about 132M characters) to BPE training. It now selects deterministic, evenly spaced training lines up to 10M characters by default. Set `TOKENIZER_MAX_CHARS=0` to opt into the full corpus. Validation text is never used to train BPE.
2. **Position information.** The earlier model had causal attention but no explicit positional encoding (documented in the old cache note). This is a serious limitation for next-token language modeling. Learned absolute position embeddings are now added to token embeddings. Full forward, contiguous KV prefill/decode, and paged KV prefill/decode use consistent absolute positions.
3. **Immutable source config.** The runner no longer edits `configs/t4-wikipedia-pretrain.yaml`; it writes `configs/t4-wikipedia-pretrain.runtime.yaml` with the tokenizer's actual vocabulary size.
4. **Checkpoint CUDA RNG restoration.** CUDA RNG state tensors are moved to CPU before calling the CUDA RNG restoration API.
5. **Evaluation reporting.** Validation logs now separate total objective, CE, and CE-derived perplexity. Perplexity is not calculated from auxiliary routing losses.
6. **Data validation.** Empty binary token files fail with a clear error; Wikipedia tokenization fails if the required EOS token is missing; downloader split identity has a fallback for null/empty page IDs.
7. **Regression tests.** Added tests for positional embeddings and bounds, empty token files, and bounded tokenizer sampling.

## Verification performed in this environment

- `python -m compileall -q spider_moe scripts`: passed.
- `python -m pytest -q`: 33 passed, 1 skipped.
- CPU end-to-end trainer integration using generated uint32 train/validation streams: passed; best/latest/step checkpoints produced; validation output includes total loss, CE, and perplexity.

## Remaining unverified items

- Hugging Face Wikipedia streaming/download was not run in this environment.
- The `tokenizers` package was not installed here, so a real BPE training run was not executed here.
- No NVIDIA GPU is available here; Colab T4 AMP, memory use, throughput, and long-run convergence must be verified in the user's runtime.
- A successful CPU smoke test is not evidence of language quality. Real held-out generation and validation curves are still required.

## Compatibility warning

The added `position_embedding` parameter changes the model state dict. Old synthetic checkpoints and checkpoints created by the previous Milestone 2 archive do not contain it and must not be used to resume this corrected run. Start from a fresh model initialization.
