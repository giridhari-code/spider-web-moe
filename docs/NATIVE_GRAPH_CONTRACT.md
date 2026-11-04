# Native Graph Contract (Spider sparse message passing)

## Purpose

Align PyTorch `SpiderMoELayer` edge construction with the native CSR kernels.

## Dimensions

| Symbol | Meaning |
|--------|---------|
| `B` | Tokens (batch × seq flattened) |
| `H` | Hidden size |
| `E` | Experts |
| `K` | Router top-k |
| `M` | Edges per expert |

## Phase 1.1 semantics (matches PyTorch)

```text
W_edge[e]: [H_out, H_in]
projected[b, e, :] = h[b] @ W_edge[e].T

for src in router_top_k(token):
  for edge (src -> dst) with weight ew:
    messages[b] += ew * projected[b, dst, :]
```

- Source set is **binary** top-k (not route softmax weights).
- Route weights apply in expert FFN, not graph.
- Use `spider_edge_project_f32` + `spider_graph_message_passing_v2_f32`.

## Phase 1 legacy (still available)

```text
messages[b] += edge_w * route_w * hidden[b]
```

`spider_graph_message_passing_f32` — for older tests only.

## CSR layout

```
row_ptr: int32[E+1]
col_idx: int32[nnz]
edge_w:  float[nnz]  # or [num_loops, nnz]
```

## Validation

- Identity W_edge → dest projection equals h
- graph_v2 with dest expert picks projected[dst]
- scalar vs NumPy max abs error < 1e-4
