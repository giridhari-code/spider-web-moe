# Native Graph Contract (Spider sparse message passing)

## Purpose

Align PyTorch `SpiderMoELayer` edge construction with the native CSR kernels so
scalar/CUDA/ARM paths are numerically definable and testable.

## Dimensions

| Symbol | Meaning |
|--------|---------|
| `B` | Batch × sequence flattened tokens (Phase 1 treats each token independently) |
| `H` | Hidden size |
| `E` | Number of experts |
| `K` | Router top-k (active source experts per token) |
| `M` | Edges per expert (`edges_per_expert`) |
| `L` | Loop index `0 .. num_loops-1` |

## Router outputs (per token)

- `expert_assign`: `int32[B, K]` — expert ids in `[0, E)` or `-1` for unused
- `route_weights`: `float[B, K]` — renormalized weights, sum to 1 over valid k

## CSR adjacency

```
row_ptr: int32[E+1]     // row_ptr[0]=0, row_ptr[E]=nnz
col_idx: int32[nnz]     // destination expert id
edge_w:  float[nnz]     // edge weight after softmax
```

**Phase 1 native reference semantics (matches scalar.cpp):**

For each token `b`, each selected source `src = assign[b,k]`:

```
for edge in [row_ptr[src], row_ptr[src+1]):
    scale = edge_w[edge] * route_weights[b,k]
    messages[b, :] += scale * hidden[b, :]
```

### PyTorch mapping note

PyTorch applies per-expert `edge_value` linear maps before gather. Native Phase 1
does **not** yet include that. Full torch parity needs Phase 1.1 (pre-apply edge_value
or extend CSR kernel). Use this contract for native↔native parity until then.

## Building CSR from edge top-m

```python
row_ptr = [0]; col_idx, edge_w = [], []
for src in range(E):
    for m in range(M):
        col_idx.append(int(edge_indices[src, m]))
        edge_w.append(float(edge_weights[src, m]))
    row_ptr.append(len(col_idx))
```

Phase 1 Orbit assumes per-loop static CSR exported with weights.

## Validation

- row_ptr[0]==0, row_ptr[E]==nnz
- col_idx in [0,E), assign in [-1,E)
- scalar vs CUDA max abs error < 1e-4 on FP32 (H≤256, B≤32)
