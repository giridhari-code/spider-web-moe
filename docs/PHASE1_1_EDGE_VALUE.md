# Phase 1.1 — edge_value fusion + full MoE loop compose

## Goal
Close the numerical gap between PyTorch `SpiderMoELayer` graph path and native kernels.

## PyTorch reference

```text
W_edge[e] : [H_out, H_in]
transformed[b, e, :] = h[b] @ W_edge[e].T

For each token b:
  messages[b] = 0
  for each source expert src in router top-k:
    for each edge (src -> dst) with weight ew:
      messages[b] += ew * transformed[b, dst, :]
```

- Graph uses **binary** source activity (top-k), not softmax route weights.
- Route weights apply in expert FFN combine.
- Projection indexed by **destination** expert id.

## New API

- `spider_edge_project_f32(hidden, B, H, E, W_edge, out_proj)`
- `spider_graph_message_passing_v2_f32(projected, ..., assign, messages)`
- `spider_moe_loop_step_f32(...)` full compose (returns 0, not -3)

Legacy `spider_graph_message_passing_f32` kept for Phase-1 tests.

## Build

Add `kernels/scalar/scalar_phase1_1.cpp` to `spider_kernels` target.
Add test `kernels/tests/test_phase1_1.cpp`.

See `CMakeLists_snippet.md` in package / commit message.
