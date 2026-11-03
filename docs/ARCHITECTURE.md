# 🕸️ Spider-Web Looped MoE — Architecture

> Complete architecture specification for a personal-scale LLM (100M–1B params)
> combining **Transformer + Mixture-of-Experts + Looped Depth + Dynamic Graph Routing**.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Design Goals](#2-design-goals)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Spider-Web MoE Layer](#5-spider-web-moe-layer)
6. [Looping Mechanism](#6-looping-mechanism)
7. [Graph Routing](#7-graph-routing)
8. [Stability Design](#8-stability-design)
9. [Early Exit](#9-early-exit)
10. [Training vs Inference](#10-training-vs-inference)
11. [Parameter Budget](#11-parameter-budget)
12. [Design Trade-offs](#12-design-trade-offs)
13. [Comparison with Baselines](#13-comparison-with-baselines)
14. [Open Questions](#14-open-questions)
15. [References](#15-references)

---

## 1. Vision

Ek **compute-efficient** LLM jo chhote parameter budget mein badi model jaisi capability de. Teen orthogonal scaling axes ko combine karta hai:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   CAPACITY    ──▶  Mixture-of-Experts               │
│                    (parameters ↑, compute →)        │
│                                                     │
│   DEPTH       ──▶  Looping                          │
│                    (effective depth ↑, params →)    │
│                                                     │
│   DIVERSITY   ──▶  Spider-Web Graph                 │
│                    (per-loop different paths)       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Core insight:**
- Dense models → compute ∝ params
- MoE → capacity ↑ without compute ↑
- Looping → depth ↑ without params ↑
- Graph → diversity ↑ without either ↑

Ye teeno alag-alag dimensions hain — inhe combine karne se **Pareto frontier** shift hoti hai.

---

## 2. Design Goals

| Goal | Target | Why |
|---|---|---|
| **Parameter efficiency** | 400M total, 150M active | Single consumer GPU (RTX 4090) |
| **Effective depth** | 20+ layers (2 + 6×3 + 2) | Small params, deep compute |
| **Training stability** | No divergence up to r=9 | LOOM criteria |
| **Inference speed** | 2–3× faster than dense | Early exit + MoE |
| **Expert utilization** | > 60% per expert | Anti-collapse |
| **Personal-scale** | ≤ 8GB VRAM trainable | LoRA + gradient checkpointing |

**Non-goals:**
- Frontier-scale (100B+) training
- Multimodal
- Multi-node distributed

---

## 3. High-Level Architecture

```
        ┌─────────────────────────────────┐
        │      INPUT TOKENS  [B, T]       │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   TOKEN EMBEDDING               │
        │   E ∈ ℝ^(V × H)                 │
        └────────────────┬────────────────┘
                         │  x₀
        ╔════════════════▼════════════════╗
        ║      PRELUDE                    ║
        ║      2 unshared Transformer     ║
        ║      layers                     ║
        ║      (unique params)            ║
        ╚════════════════╤════════════════╝
                         │  h₀
        ╔════════════════▼════════════════╗
        ║   RECURRENT BLOCK               ║
        ║                                 ║
        ║   SpiderMoE layer × r loops     ║
        ║   (shared params)               ║
        ║                                 ║
        ║   ┌───────────────────────┐     ║
        ║   │  t=1 → t=2 → … → t=r  │     ║
        ║   │  each loop:           │     ║
        ║   │   router → edgegen    │     ║
        ║   │   → dispatch → FFN    │     ║
        ║   │   → epilogue          │     ║
        ║   │   → early exit?       │     ║
        ║   └───────────────────────┘     ║
        ║                                 ║
        ╚════════════════╤════════════════╝
                         │  h_r
        ╔════════════════▼════════════════╗
        ║      CODA                       ║
        ║      2 unshared Transformer     ║
        ║      layers                     ║
        ║      (unique params)            ║
        ╚════════════════╤════════════════╝
                         │  z
                         ▼
        ┌─────────────────────────────────┐
        │   LM HEAD   logits = z · Wₒ     │
        │   [B, T, V]                     │
        └─────────────────────────────────┘
```

**Layer count (effective depth):**

| Section | Layers | Loops | Effective |
|---|---|---|---|
| Prelude | 2 | 1 | 2 |
| Recurrent | 4 | 6 | 24 |
| Coda | 2 | 1 | 2 |
| **Total** | **8** | — | **28** |

Baseline dense model with 28 layers → ~1.5B params.
Spider-Web → ~400M params, same effective depth. **~4× smaller**.

---

## 4. Component Breakdown

### 4.1 Prelude

```
┌────────────────────────────────────┐
│  PRELUDE LAYER (×2)                │
│                                    │
│   h ← h + Attn(LN(h))              │
│   h ← h + FFN(LN(h))               │
│                                    │
│   • Standard Transformer layer     │
│   • Unique params per layer        │
│   • NOT looped                     │
└────────────────────────────────────┘
```

**Why separate?**
- Root-level token processing
- Stabilizes input before looping
- Huginn-style: unlooped entry prevents variance issues early
- Cheap: only 2 layers

### 4.2 Recurrent Block

```
┌────────────────────────────────────┐
│  SPIDER-MoE LAYER (shared, ×r)     │
│                                    │
│   for t in 1..=r:                  │
│       h = spider_moe(h, x₀, t)     │
│       if early_exit(h): break      │
│                                    │
│   • SAME params across loops       │
│   • loop index t is input signal   │
│   • graph evolves per loop         │
└────────────────────────────────────┘
```

**Why loop?**
- Params shared → memory efficient
- Effective depth without param cost
- Gradient flows through all loops (BPTT)
- Loop count can vary at inference (early exit)

### 4.3 Coda

```
┌────────────────────────────────────┐
│  CODA LAYER (×2)                   │
│                                    │
│   h ← h + Attn(LN(h))              │
│   h ← h + FFN(LN(h))               │
│                                    │
│   • Standard Transformer layer     │
│   • Unique params per layer        │
│   • NOT looped                     │
└────────────────────────────────────┘
```

**Why separate?**
- Output-side refinement
- Symmetric with Prelude
- Provides "clean exit" from loops

---

## 5. Spider-Web MoE Layer

Ek loop iteration ka complete flow:

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   Input:  h_{t-1},  x₀,  t                                 │
│                                                            │
│   ┌────────────────────────────────────────────────┐       │
│   │ ① ROUTER                                       │       │
│   │    logits = h_{t-1} W_r + b_r[t]               │       │
│   │    probs  = softmax(logits)                    │       │
│   │    (idx, w) = TopK(probs, k)                   │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │  idx, w                               │
│                    ▼                                       │
│   ┌────────────────────────────────────────────────┐       │
│   │ ② EDGE GENERATOR                               │       │
│   │    e_t   = LoopEmbed(t)                        │       │
│   │    g_t   = σ([h_{t-1}; e_t] W_g)               │       │
│   │    A_t   = SparseTopK(g_t, m)                  │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │  A_t (CSR sparse)                     │
│                    ▼                                       │
│   ┌────────────────────────────────────────────────┐       │
│   │ ③ SPARSE DISPATCH                              │       │
│   │    m_t = Σ α_ij E_j(h_{t-1})                   │       │
│   │           over (i,j) ∈ A_t                     │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │  m_t                                  │
│                    ▼                                       │
│   ┌────────────────────────────────────────────────┐       │
│   │ ④ EXPERT FFN (Grouped GEMM)                    │       │
│   │    Δ_t = Σ_k w_k FFN_{idx_k}(m_t)              │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │  Δ_t                                  │
│                    ▼                                       │
│   ┌────────────────────────────────────────────────┐       │
│   │ ⑤ LOOP EPILOGUE                                │       │
│   │    u_t = h_{t-1} + α Δ_t + β x₀                │       │
│   │    h_t = LayerNorm(u_t)                        │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │  h_t                                  │
│                    ▼                                       │
│   ┌────────────────────────────────────────────────┐       │
│   │ ⑥ EARLY EXIT (inference only)                  │       │
│   │    c_t = σ(h_t · w_exit)                       │       │
│   │    if c_t > τ: return h_t                      │       │
│   └────────────────┬───────────────────────────────┘       │
│                    │                                       │
│                    ▼                                       │
│   Output: h_t                                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Param sharing strategy:**

| Component | Shared across loops? | Why |
|---|---|---|
| Router weights | ✅ (with per-loop bias) | Symmetry broken by `b_r[t]` |
| Edge generator | ✅ (with loop embed) | Loop index is input |
| Expert FFN | ✅ | Main capacity |
| Epilogue (α, β) | ✅ | Stability constants |
| Attention | ❌ (untied) | Foil pattern |
| LayerNorm | ✅ | Standard |

---

## 6. Looping Mechanism

### 6.1 Loop Count Trade-off

```
Loop count r:     3     5     7     9     12
                  │     │     │     │     │
Stability         ✅    ✅    ✅    ✅    ⚠️
Perplexity gain   —     ↑↑    ↑↑↑   ↑↑↑↑  saturates
VRAM (activation) low   med   med   high  very high
Inference speed   fast  med   slow  slower slowest
```

**Recommendation:** `r = 6` for personal-scale (best balance).

### 6.2 Loop Evolution

```
Loop 1:  h₀ → [Router₁, Graph₁, FFN] → h₁
Loop 2:  h₁ → [Router₂, Graph₂, FFN] → h₂
Loop 3:  h₂ → [Router₃, Graph₃, FFN] → h₃
...
Loop r:  h_{r-1} → [Router_r, Graph_r, FFN] → h_r

         Same weights, different:
         • Router bias b_r[t]
         • Graph edges A_t
         • Loop embed e_t
```

### 6.3 What Changes Per Loop?

```
┌──────────────────┬─────────────────┬───────────────────┐
│  Component       │  Same?          │  Different?       │
├──────────────────┼─────────────────┼───────────────────┤
│  Router W_r      │  ✅             │  bias b_r[t]      │
│  Edge gen W_g    │  ✅             │  loop embed e_t   │
│  Expert FFN      │  ✅             │  activation       │
│  Attention       │  ❌ untied      │  full params      │
│  LayerNorm       │  ✅             │  —                │
│  Residual α, β   │  ✅             │  —                │
└──────────────────┴─────────────────┴───────────────────┘
```

---

## 7. Graph Routing

### 7.1 The Problem with Vanilla MoE

```
Normal MoE:  Token → Router → top-k experts → output

Problem: Router is "stateless" — same input → same experts.
         In looping, this causes expert selection collapse.
```

### 7.2 Spider-Web Solution

```
Spider-Web:  Token → Router → experts
                        ↓
                     Edges (learned)
                        ↓
                  Next experts → output

Edges are:
• Sparse (3 per expert)
• Loop-dependent (change per t)
• Learned (via gradient)
```

### 7.3 Visual Comparison

```
Normal MoE:                    Spider-Web MoE:

  ┌─────────┐                   ┌─────────┐
  │ Router  │                   │ Router  │
  └────┬────┘                   └────┬────┘
   ┌───┼───┐                    ┌────┴────┐
   ▼   ▼   ▼                    ▼         ▼
 ┌──┐ ┌──┐ ┌──┐               ┌──┐       ┌──┐
 │E1│ │E2│ │E3│               │E1│◀─────▶│E3│
 └──┘ └──┘ └──┘               └──┬─┘     └──┬─┘
                                  │  ╲   ╱  │
    (isolated)                    ▼   ╲ ╱   ▼
                                ┌──┐  ╳  ┌──┐
                                │E2│◀───▶│E4│
                                └──┘     └──┘

                              (connected graph)
```

### 7.4 Per-Loop Edge Sets

```
Loop 1:               Loop 2:               Loop 3:
  E1 ──▶ E3             E1     E3             E1 ──▶ E2
  E2 ──▶ E4             E2 ──▶ E1             E2     E4
                        E3 ──▶ E4               │
                                                ▼
                                               E3

Different active edges per loop
→ different computational paths
→ no expert collapse
```

### 7.5 Graph Representation

```
Adjacency (CSR format):
  row_ptr = [0, 3, 6, 9, 12]   (offsets per expert)
  col_idx = [2, 5, 7, 1, 3, 6, ...]   (neighbor ids)
  weights = [0.5, 0.3, 0.2, ...]      (edge weights)

Per-loop weights:
  edge_weights[t] = [num_experts, edges_per_expert]

Memory: E × m × r × 4 bytes = 8 × 3 × 6 × 4 = 576 bytes
        → negligible
```

---

## 8. Stability Design

### 8.1 The Problems

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Problem 1: Variance blow-up                           │
│  ────────────────────────                              │
│  Each loop adds Δ → variance grows exponentially       │
│  h_t variance ∝ 1.2^t → explode at t=10                │
│                                                        │
│  Problem 2: Representation drift                       │
│  ────────────────────────────                          │
│  h_t moves far from x₀ → gradient signal lost          │
│  Model forgets input                                   │
│                                                        │
│  Problem 3: Weight-sharing symmetry                    │
│  ────────────────────────────────                      │
│  Same W applied → same output                          │
│  Loop produces no new computation                      │
│                                                        │
│  Problem 4: Expert collapse                            │
│  ────────────────────────                              │
│  Router picks same experts every loop                  │
│  Extra loops = wasted compute                          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 8.2 The Fixes

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Fix 1: Residual scaling                               │
│  ────────────────────────                              │
│  u_t = h_{t-1} + α · Δ_t                               │
│  α = 0.5 → bounded variance growth                     │
│                                                        │
│  Fix 2: Input re-injection                             │
│  ────────────────────────────                          │
│  u_t = ... + β · x₀                                    │
│  β = 0.1 → anchored to input                           │
│                                                        │
│  Fix 3: Per-loop bias                                  │
│  ────────────────────────                              │
│  logits = h · W_r + b_r[t]                             │
│  b_r[t] unique per loop → breaks symmetry              │
│                                                        │
│  Fix 4: Dynamic graph                                  │
│  ────────────────────────                              │
│  A_t = SparseTopK(g_t, m)                              │
│  Graph changes per loop → new paths                    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 8.3 Stability Monitor (training)

```
Track per-loop:
  • hidden state variance  Var(h_t)
  • mean hidden norm       ‖h_t‖₂
  • gradient norm          ‖∇h_t‖₂
  • expert load histogram  count(experts)

Alert if:
  Var(h_t) > 10 × Var(h_0)      → residual scaling off
  ‖h_t‖₂ > 5 × ‖h_0‖₂            → drift
  max(expert_load) > 5 × mean   → collapse
```

---

## 9. Early Exit

### 9.1 Concept

```
Loop 1:  conf=0.3  → continue
Loop 2:  conf=0.5  → continue
Loop 3:  conf=0.7  → continue
Loop 4:  conf=0.9  → EXIT ✅ (saved 2 loops)
```

### 9.2 Architecture

```
┌────────────────────────────────────────────┐
│  At each loop boundary (after epilogue):   │
│                                            │
│   conf = σ(h_t · w_exit)    ← learned      │
│                                            │
│   if conf > τ:                             │
│       return h_t                           │
│   else:                                    │
│       continue to next loop                │
└────────────────────────────────────────────┘
```

### 9.3 Per-Token Exit

```
Batch of 4 tokens:
  token 1:  exits at loop 3  (conf=0.87)
  token 2:  exits at loop 6  (conf=0.72)
  token 3:  exits at loop 4  (conf=0.91)
  token 4:  exits at loop 6  (conf=0.68)

→ Mask-based: active_tokens maintained per loop
→ Saves ~30% compute on average
```

### 9.4 When to Use

```
┌──────────────────┬──────────┬───────────┐
│  Phase           │  Exit?   │  Why      │
├──────────────────┼──────────┼───────────┤
│  Training        │  ❌ NO   │  Breaks   │
│                  │          │  gradient │
│                  │          │  flow     │
├──────────────────┼──────────┼───────────┤
│  Inference       │  ✅ YES  │  Speed    │
│                  │          │  matters  │
├──────────────────┼──────────┼───────────┤
│  Evaluation      │  ✅ YES  │  Matches  │
│                  │          │  inference│
└──────────────────┴──────────┴───────────┘
```

---

## 10. Training vs Inference

### 10.1 Training Flow

```
┌──────────────────────────────────────────────┐
│  TRAINING ITERATION                          │
├──────────────────────────────────────────────┤
│                                              │
│  Data Loader → batch [B, T]                  │
│         │                                    │
│         ▼                                    │
│  Forward pass:                               │
│    • Full r loops (NO early exit)            │
│    • Track per-loop variance                 │
│    • Accumulate load balance stats           │
│         │                                    │
│         ▼                                    │
│  Loss = CE + λ₁·aux + λ₂·edge                │
│         │                                    │
│         ▼                                    │
│  Backward pass (BPTT through loops):         │
│    • Gradient flows through all r loops      │
│    • Gradient checkpointing mandatory        │
│         │                                    │
│         ▼                                    │
│  Optimizer step:                             │
│    • AdamW, lr=3e-4, cosine schedule         │
│    • Grad clip norm=1.0                      │
│    • Expert load balancing                   │
│         │                                    │
│         ▼                                    │
│  Checkpoint (safetensors)                    │
│                                              │
└──────────────────────────────────────────────┘
```

### 10.2 Inference Flow

```
┌──────────────────────────────────────────────┐
│  INFERENCE (autoregressive)                  │
├──────────────────────────────────────────────┤
│                                              │
│  Prefill:                                    │
│    • Process prompt (full r loops)           │
│    • Build KV cache                          │
│    • No early exit (need full context)       │
│         │                                    │
│         ▼                                    │
│  Decode (per token):                         │
│    • Loop through r iterations               │
│    • Early exit check per token              │
│    • Sample from h_exit                      │
│    • Append to KV cache                      │
│         │                                    │
│         ▼                                    │
│  Output: tokens                              │
│                                              │
│  Optimizations:                              │
│    • CUDA graphs (loop capture)              │
│    • Paged KV cache (loop-aware)             │
│    • Continuous batching                     │
│    • Speculative decoding (future)           │
│                                              │
└──────────────────────────────────────────────┘
```

### 10.3 Key Differences

| Aspect | Training | Inference |
|---|---|---|
| Loops | Always r | Up to r (early exit) |
| Batch | Full [B, T] | Continuous batching |
| KV cache | Recomputed | Paged, cached |
| Grad | Flows through loops | No grad |
| Precision | FP32 accum | FP16/BF16 |
| Graph | Static per loop | Dynamic (per token) |

---

## 11. Parameter Budget

### 11.1 Breakdown (H=512, E=8, r=6)

```
┌────────────────────────┬──────────────┬─────────────┐
│  Component             │  Params      │  Notes      │
├────────────────────────┼──────────────┼─────────────┤
│  Embedding             │  16.4M       │  32K × 512  │
│                        │              │             │
│  Prelude (2 layers)    │  6.3M        │  2 × 3.1M   │
│    • Attn (4 heads)    │  1.0M        │  per layer  │
│    • FFN (512→2048)    │  2.1M        │  per layer  │
│                        │              │             │
│  Recurrent (shared)    │              │             │
│    • Attn (untied × r) │  6.3M        │  r × 1.0M   │
│    • Router            │  4.1K        │  per loop   │
│    • Edge gen          │  65K         │  shared     │
│    • Experts (×8)      │  16.8M       │  8 × 2.1M   │
│    • Epilogue          │  1K          │  α, β       │
│                        │              │             │
│  Coda (2 layers)       │  6.3M        │  2 × 3.1M   │
│                        │              │             │
│  LM Head               │  16.4M       │  tied w/emb │
│                        │              │             │
├────────────────────────┼──────────────┼─────────────┤
│  Total stored          │  ~68M        │             │
│  Active per token      │  ~25M        │  (k=2, E=8) │
└────────────────────────┴──────────────┴─────────────┘
```

### 11.2 Scaling Up

```
For 400M params (target):
  • H = 1024
  • E = 16
  • experts per layer = 16
  • r = 6

For 1B params:
  • H = 2048
  • E = 32
  • r = 8
```

---

## 12. Design Trade-offs

### 12.1 What We Gain

```
✅ Parameter efficiency   — 4× smaller than dense equivalent
✅ Inference speed        — 2–3× faster with early exit
✅ Effective depth        — 20+ layers with 8 stored
✅ Compute efficiency     — MoE top-k keeps active small
✅ Diversity              — Dynamic graph prevents collapse
```

### 12.2 What We Sacrifice

```
❌ Training complexity    — BPTT through loops = memory heavy
❌ Kernel complexity      — Graph routing ≠ standard MoE
❌ Debug difficulty       — Loop + graph = hard to trace
❌ Convergence           — More unstable than dense
❌ Tooling               — No off-the-shelf support
```

### 12.3 Alternatives Considered

| Alternative | Why not |
|---|---|
| Dense Transformer | Too many params for same capability |
| Vanilla MoE | No depth scaling, expert collapse |
| Standard Looping (Huginn) | Dense looped doesn't scale |
| GraphMoE only | No depth scaling |
| Looped MoE (LOOM) | No graph diversity |

**Spider-Web MoE = intersection of all four.**

---

## 13. Comparison with Baselines

```
┌─────────────────────┬───────────┬──────────┬──────────┬──────────┐
│  Metric             │  Dense    │  MoE     │  Looped  │  Spider  │
│                     │  (28L)    │  (dMoE)  │  (Huginn)│  (ours)  │
├─────────────────────┼───────────┼──────────┼──────────┼──────────┤
│  Total params       │  1.5B     │  2B      │  400M    │  400M    │
│  Active params      │  1.5B     │  500M    │  400M    │  150M    │
│  Effective depth    │  28       │  8       │  28      │  28      │
│  Training stability │  ✅       │  ⚠️      │  ⚠️      │  ⚠️      │
│  Expert collapse    │  N/A      │  ⚠️      │  N/A     │  ✅      │
│  Inference speed    │  baseline │  2×      │  1×      │  3×      │
│  Memory (train)     │  high     │  high    │  med     │  med     │
│  Kernel complexity  │  low      │  med     │  low     │  high    │
└─────────────────────┴───────────┴──────────┴──────────┴──────────┘
```

**Conclusion:** Spider-Web MoE Pareto-dominates on **params × depth × speed**, at the cost of **implementation complexity**.

---

## 14. Open Questions

Ye research-grade design hai — kuch cheezein **unproven** hain:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ❓ Q1: Does dynamic graph actually help over fixed?     │
│     → Ablation needed: fixed A vs dynamic A              │
│                                                          │
│  ❓ Q2: What's optimal edges_per_expert?                 │
│     → Too few = no diversity, too many = compute blast   │
│                                                          │
│  ❓ Q3: Untied attention worth the param cost?           │
│     → Compare r×1.0M vs 1.0M shared                      │
│                                                          │
│  ❓ Q4: Does early exit hurt quality?                    │
│     → Measure perplexity with/without exit               │
│                                                          │
│  ❓ Q5: How does r affect convergence?                   │
│     → Loss curves for r = 3, 6, 9, 12                    │
│                                                          │
│  ❓ Q6: Interaction between graph and load balance?      │
│     → Does L_edge need tuning per r?                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Recommendation:** **Ablate one variable at a time.** Don't build the full thing first — validate each piece on 50M–100M scale.

---

## 15. References

1. **Huginn** — *"Scalable Recurrent Depth"* (Geiping et al., 2024)
2. **LOOM** — *"Looping Meets MoE"* (2025)
3. **Foil** — *"How to Loop MoE"* (2025)
4. **GraphMoE** — *"Dynamic Edge Generation for MoE"*
5. **SymphonySMoE** — *"Social Graph Routing"*
6. **MoE-Infinity** — Sparse MoE kernel design
7. **nano-vllm** — Inference engine structure
8. **Candle** — Rust ML framework

---

## Appendix A: File → Component Map

```
spider-moe/
├── crates/spider-core/src/
│   ├── model/prelude.rs      → Section 4.1
│   ├── model/recurrent.rs    → Section 4.2, 5, 6
│   ├── model/coda.rs         → Section 4.3
│   ├── model/early_exit.rs   → Section 9
│   ├── graph/edge_generator  → Section 7
│   ├── graph/adjacency.rs    → Section 7.5
│   ├── moe/router.rs         → Section 5
│   ├── moe/load_balance.rs   → Section 8
│   └── stability/            → Section 8
└── kernels/
    ├── router_fused.cu       → Section 5 ①
    ├── graph_message.cu      → Section 5 ③
    ├── expert_grouped_gemm   → Section 5 ④
    ├── loop_epilogue.cu      → Section 5 ⑤
    └── early_exit_gate.cu    → Section 5 ⑥
```

---

## Appendix B: Glossary

| Term | Meaning |
|---|---|
| **Prelude** | Unlooped entry layers |
| **Recurrent Block** | Looped MoE body |
| **Coda** | Unlooped exit layers |
| **Loop iteration** | One pass through the shared block |
| **Spider-web** | Dynamic sparse expert graph |
| **Edge** | Connection between two experts |
| **Router** | Expert selector per token |
| **Collapse** | Router always picks same experts |
| **Early exit** | Stop looping when confident |
| **Re-inject** | Add original input back per loop |
| **BPTT** | Backprop through time (loops) |
| **CSR** | Compressed Sparse Row format |

---

*Document version: 1.0 · Target: personal LLM (100M–1B)*
*Status: Research-grade · Ablation required before scale-up*