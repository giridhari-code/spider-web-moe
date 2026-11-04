# Phase 1.3 — Native edge_generator + ctypes

## Goal
Bring PyTorch `edge_generator: Linear(H + loop_embed → E*E)` into scalar C, and expose Phase 1.1–1.3 graph APIs via ctypes.

## C API
- `spider_edge_generator_f32` — cat(h, loop_emb) @ W_gen^T → [B,E,E]
- `spider_edge_generator_topk_f32` — generator + top-M softmax
- `spider_graph_from_generator_f32` — generator → topk → edge_value project → dynamic gather

## Python
`NativeKernels`: `edge_project`, `graph_v2`, `edge_topk`, `edge_generator`, `graph_dynamic`, `graph_from_generator` (NumPy fallbacks if symbols missing).

## Build
Link `kernels/scalar/scalar_phase1_3.cpp`. Test `spider_kernel_phase1_3`.
