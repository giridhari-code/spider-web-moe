# Phase 1.2 — Dynamic per-token graph edges

## Goal

Match PyTorch `SpiderMoELayer` **edge_generator** path:

```text
edge_logits = Linear(cat(h, loop_emb)) → [B, E, E]
edge_indices, edge_weights = topk + softmax over dest (M)
projected[b,e] = h[b] @ W_edge[e].T
for src in router_top_k:
  for m in 0..M-1:
    messages += edge_weights[b,src,m] * projected[b, edge_indices[b,src,m]]
```

Static CSR (Phase 1 / 1.1) still supported.

## API

| Function | Role |
|----------|------|
| `spider_edge_topk_f32` | top-M + softmax per source row |
| `spider_graph_message_passing_dynamic_f32` | per-token edges + projected dest |
| `spider_graph_dynamic_with_edge_value_f32` | project + dynamic graph |

Layouts: `edge_logits [B,E,E]`, `edge_indices/weights [B,E,M]`, `projected [B,E,H]`, `assign [B,K]`.

## Build

Link `kernels/scalar/scalar_phase1_2.cpp`; test `spider_kernel_phase1_2`.
