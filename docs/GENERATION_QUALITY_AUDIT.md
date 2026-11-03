# Generation Quality Audit — Wikipedia Pretraining

## Confirmed issue: untrained early-exit head used during cached generation

The language-model training path calls `model(x, ...)` with `early_exit=False` by default and trains all configured recurrent loops. The generation path prefills with all loops, but incremental decode previously defaulted to `early_exit=True` in `decode_step()` and `generate()`. The recurrent `exit_head` is not supervised by the LM loss, so its confidence values are not calibrated. As a result, cached generation could execute fewer recurrent loops than training, creating a train/inference mismatch and unstable generated text.

### Fix

- `decode_step(..., early_exit=False)` by default.
- `decode_step_paged(..., early_exit=False)` by default.
- `generate(..., early_exit=False)` by default.
- `generate_paged(..., early_exit=False)` by default.
- Early exit remains explicitly opt-in for experiments after a proper exit-head training objective and calibration are implemented.

## Gradient log correction

Gradient accumulation intentionally performs no optimizer update on intermediate microbatches. The previous logger printed `0.000` on those microbatches, which could be mistaken for zero gradients. It now prints the most recent actual optimizer-update gradient norm, explicitly labeled `(last update)`.

## Validation

- Added regression test forcing the exit head to request immediate exit and verifying default cached decode still matches full-depth forward.
- Test suite: 34 passed, 1 skipped (CUDA/Triton hardware test).

## Interpretation of the supplied training run

The run processed 2,000 optimizer steps with batch size 2, sequence length 256 and accumulation 8: approximately 8.192 million training tokens (not 35.54 million). The token file size is the available corpus, not the amount consumed by this run. Validation CE improved from 8.7881 at step 100 to 7.2610 at step 2000, but this is still an early, weak language model. The generation mismatch is a confirmed bug; the remaining high perplexity also indicates the model needs substantially more training and/or improved model/data capacity after inference is corrected.

## Not claimed

This audit does not establish that the model is fluent or production-ready. It does not validate CUDA/Triton kernels on the user's T4. Positional embeddings are present in this project revision; do not confuse this with proof that all architecture choices are optimal.
