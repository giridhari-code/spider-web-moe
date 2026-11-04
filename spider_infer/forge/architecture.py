"""Forge architecture blueprint (no missing pillars).

Layer cake
----------
  Client / HTTP
       │
  Orbit  (serve: batching, KV shells, graphs, draft orbits)
       │
  Dobu   (weight chunk residency: CPU catalogue → GPU LRU)
       │
  **Forge** (this fabric)
       ├─ attention  ← FlashMLA
       ├─ gemm       ← DeepGEMM
       ├─ ep         ← DeepEP
       ├─ tile       ← TileLang
       └─ select     ← DeepSelect
       │
  Spider / Sentinel weights + kernels backends (torch → triton → cuda)

Design rules
------------
1. Every DeepInfer vendor capability has a Forge pillar (API always exists).
2. Reference TORCH backend always works; fast paths plug in behind the same calls.
3. top_k (select/router) ≠ top_dobu (Dobu residency); Forge Select exposes both.
4. EP combine is mandatory for multi-rank MoE; single-rank is a no-op combine.
5. Tile specs decouple block sizes from Python MoE loops.
6. Orbit never imports FlashMLA/DeepGEMM symbols directly — only Forge.*.

Roadmap of bodies (API frozen)
------------------------------
  0.1  torch reference (shipped skeleton)
  0.2  Triton paged decode + GEMM expert path
  0.3  CUDA FlashMLA-style + DeepGEMM FP8
  0.4  DeepEP-style dispatch (replace all_reduce where beneficial)
  0.5  Ascend / multi-backend
"""

FORGE_PILLARS = (
    "attention",  # FlashMLA
    "gemm",       # DeepGEMM
    "ep",         # DeepEP
    "tile",       # TileLang
    "select",     # DeepSelect
)

DEEPINFER_MAP = {
    "FlashMLA": "forge.attention",
    "DeepGEMM": "forge.gemm",
    "DeepEP": "forge.ep",
    "TileLang": "forge.tile",
    "DeepSelect": "forge.select",
}
