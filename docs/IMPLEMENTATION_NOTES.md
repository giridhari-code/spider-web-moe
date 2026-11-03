# Implementation notes and deviations

This code follows the supplied Spider-Web Looped MoE specifications as a runnable baseline.

## Implemented
- Prelude and Coda transformer stacks.
- Shared recurrent Spider-MoE block applied `num_loops` times.
- Per-loop router bias, top-k expert selection, normalized routing weights.
- Loop embedding and learned top-m outgoing edges per source expert.
- Sparse edge gather/aggregation; no dense adjacency matrix multiply.
- Expert-specific FFNs and weighted top-k output.
- Residual scaling and original-input reinjection followed by LayerNorm.
- Inference-only early exit option, next-token CE, router/edge auxiliary terms.
- uint32 token dataset, AdamW training, clipping, checkpointing and CLI.

## Not implemented
- Custom CUDA, ARM NEON/SVE, scalar FFI kernel suite.
- Rust workspace / C++ dispatch layer.
- Paged KV cache, continuous batching, CUDA Graph capture, serving scheduler.
- Production tokenizer, SFT/DPO data adapters, HTTP server.
- Exact target parameter budget/performance claims.

## Design note
The source docs define the high-level edge generator and sparse message equation, but do not fully specify all tensor-level implementation details. This reference interprets the edge generator as per-token source-to-destination edge scores and uses destination-specific linear transforms before weighted aggregation. This choice is explicit and replaceable.
