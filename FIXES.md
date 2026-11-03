# Training stability fixes

## Confirmed defect

`nn.Embedding` was previously created without an explicit initialization. PyTorch's
default embedding initialization has standard deviation near 1.0. Since
`lm_head.weight` is tied to `token_embedding.weight`, this produced very large
initial output logits. In the reported tiny run, the embedding standard deviation
was 1.001 and one token dominated the softmax from the pretraining checkpoint.

The embedding now uses `ModelConfig.embedding_init_std` (default 0.02). A fresh
randomly initialized 256-token/64-hidden test produced logit standard deviation
about 0.156, maximum absolute logit about 0.625, and initial CE about 5.55, rather
than the pathological tens-of-logits scale.

## Additional training safeguards

- Validate model target shape and allow `-100` ignored labels in model-level loss.
- Fail clearly on all-ignored target batches and out-of-vocabulary targets.
- Validate labels shape/range in the shared LM training loss.
- Average gradient accumulation by the actual microbatch count in each group,
  including a short final group and a resumed run.
- Stop on non-finite loss instead of silently skipping a microbatch while retaining
  potentially inconsistent accumulation state.

## Validation performed

- `python -m pytest -q`: 30 passed, 1 skipped.
- `python -m compileall -q spider_moe scripts`: passed.
- CPU training smoke test: 5 steps, validation at steps 2 and 4, latest and best
  checkpoints written; embedding standard deviation remained approximately 0.02.
- CUDA/T4 and RTX execution has not been verified in this environment.

## Important migration instruction

Old checkpoints contain the old, large embedding weights. Loading one of them
restores those weights and therefore bypasses the new initialization. Start a new
pretraining run from a freshly constructed model. Do not use the old pretrain,
midtrain, SFT, or DPO checkpoints as the initial weights for the corrected run.
