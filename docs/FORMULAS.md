# 🕸️ Spider-Web Looped MoE — Visual Diagrams & Formulas

> Complete reference: architecture, formulas, kernels, and design rationale.
> Personal LLM target: 100M–1B parameters.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Loop Iteration Detail](#2-loop-iteration-detail)
3. [Spider-Web Graph Concept](#3-spider-web-graph-concept)
4. [Core Formulas](#4-core-formulas)
5. [Stability Rationale](#5-stability-rationale)
6. [Dimension Reference](#6-dimension-reference)
7. [Kernel ↔ Formula Map](#7-kernel--formula-map)
8. [End-to-End Symbolic Flow](#8-end-to-end-symbolic-flow)
9. [Default Config](#9-default-config)
10. [Full Stack](#10-full-stack)
11. [Cheat Sheet](#11-cheat-sheet)

---

## 1. Architecture Overview

```
        ┌─────────────────────────────────┐
        │      INPUT TOKENS  [B, T]       │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   EMBEDDING  E ∈ ℝ^(V×H)        │
        │   x₀ = E[token]                 │
        └────────────────┬────────────────┘
                         │
        ╔════════════════▼════════════════╗
        ║      PRELUDE  (2 layers)        ║
        ║   h₀ = Prelude(x₀)              ║
        ╚════════════════╤════════════════╝
                         │
        ╔════════════════▼════════════════╗
        ║   RECURRENT BLOCK  (r loops)    ║
        ║                                 ║
        ║   for t = 1 … r:                ║
        ║       hₜ = SpiderMoE(hₜ₋₁, t)   ║
        ║       if exit(hₜ): break        ║
        ║                                 ║
        ╚════════════════╤════════════════╝
                         │  h_r
        ╔════════════════▼════════════════╗
        ║       CODA  (2 layers)          ║
        ║   z = Coda(h_r)                 ║
        ╚════════════════╤════════════════╝
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   LM HEAD   logits = z · Wₒ     │
        │   [B, T, V]                     │
        └─────────────────────────────────┘
```

**Design principles:**

| Principle | Source | Purpose |
|---|---|---|
| Prelude–Recurrent–Coda | Huginn | Stable looping, easy early exit |
| Per-loop routers | LOOM | Symmetry breaking |
| Dynamic edge generator | GraphMoE | Anti expert-collapse |
| Flattened experts | Foil | Bigger routing pool |
| Untied attention per loop | Foil | Balanced routing |
| Residual scaling + re-inject | LOOM | Variance bound, no drift |
| Early exit gate | — | Inference speedup |

---

## 2. Loop Iteration Detail

```
┌──────────────────────────────────────────────────────────┐
│              LOOP STEP t  (hₜ₋₁ → hₜ)                    │
└──────────────────────────────────────────────────────────┘

  Input:  hₜ₋₁ ∈ ℝ^(B×H),  x₀ ∈ ℝ^(B×H),  loop index t


  ┌───────────────────────────────────────────────────┐
  │ ① ROUTER                                          │
  │                                                   │
  │   ℓₜ = hₜ₋₁ · Wᵣ + bᵣ[t]        ← loop bias        │
  │   pₜ = softmax(ℓₜ)                                │
  │   (idxₜ, wₜ) = TopK(pₜ, k)                        │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │ idxₜ, wₜ
                          ▼
  ┌───────────────────────────────────────────────────┐
  │ ② EDGE GENERATOR  (spider web)                    │
  │                                                   │
  │   eₜ = Embed(t) ∈ ℝ^(B×d)                         │
  │   gₜ = σ([hₜ₋₁ ; eₜ] · W_g)      ← edge logits    │
  │   Aₜ = SparseTopK(gₜ, m)         ← adjacency      │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │ Aₜ  (CSR sparse)
                          ▼
  ┌───────────────────────────────────────────────────┐
  │ ③ SPARSE DISPATCH  (graph message passing)        │
  │                                                   │
  │   mₜ = Σ_{(i,j)∈Aₜ} αᵢⱼ · Eⱼ(hₜ₋₁)                │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │ mₜ
                          ▼
  ┌───────────────────────────────────────────────────┐
  │ ④ EXPERT FFN  (grouped GEMM)                      │
  │                                                   │
  │   Δₜ = Σ_{k=1..K} wₜ,k · FFN_{idxₜ,k}(mₜ)         │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │ Δₜ
                          ▼
  ┌───────────────────────────────────────────────────┐
  │ ⑤ LOOP EPILOGUE  (stability)                      │
  │                                                   │
  │   uₜ = hₜ₋₁ + α · Δₜ + β · x₀   ← re-inject       │
  │   hₜ = LayerNorm(uₜ)                              │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │ hₜ
                          ▼
  ┌───────────────────────────────────────────────────┐
  │ ⑥ EARLY EXIT GATE  (inference only)               │
  │                                                   │
  │   cₜ = σ(hₜ · w_exit)                             │
  │   if cₜ > τ  →  return hₜ  (break loop)           │
  │                                                   │
  └───────────────────────┬───────────────────────────┘
                          │
                          ▼
                    Output: hₜ
```

---

## 3. Spider-Web Graph Concept

### 3.1 Normal MoE — parallel, no edges

```
                  ┌─────────┐
                  │ Router  │
                  └────┬────┘
          ┌───────┬────┴────┬───────┐
          ▼       ▼         ▼       ▼
        ┌───┐   ┌───┐     ┌───┐   ┌───┐
        │E1 │   │E2 │     │E3 │   │E4 │
        └───┘   └───┘     └───┘   └───┘
          ✗       ✓         ✗       ✓

     Experts isolated — no message passing
```

### 3.2 Spider-Web MoE — learned sparse edges

```
                  ┌─────────┐
                  │ Router  │
                  └────┬────┘
                ┌──────┴──────┐
                ▼             ▼
             ┌─────┐       ┌─────┐
             │ E1  │◀─────▶│ E3  │
             └──┬──┘       └──┬──┘
                │ ╲         ╱ │
                │  ╲       ╱  │
                │   ╲     ╱   │
                ▼    ╲   ╱    ▼
             ┌─────┐  ╲ ╱  ┌─────┐
             │ E2  │◀──╳──▶│ E4  │
             └─────┘  ╱ ╲  └─────┘

     Edges = learned, sparse, per-loop
```

### 3.3 Loop evolution — edges change each loop

```
  Loop 1:          Loop 2:          Loop 3:
  ┌───┐  ┌───┐     ┌───┐  ┌───┐     ┌───┐  ┌───┐
  │E1 │─▶│E3 │     │E1 │  │E3 │     │E1 │─▶│E2 │
  └───┘  └───┘     └─┬─┘  └───┘     └───┘  └───┘
  ┌───┐  ┌───┐       │    ┌───┐     ┌───┐  ┌───┐
  │E2 │─▶│E4 │     ┌─▼─┐  │E4 │     │E3 │  │E4 │
  └───┘  └───┘     │E2 │─▶└───┘     └─┬─┘  └───┘
                   └───┘               │
                                       ▼
                                      E4

  Different active edges per loop → prevents expert collapse
```

---

## 4. Core Formulas

### 4.1 Router

$$
\ell_t = h_{t-1} W_r + b_r[t]
$$

$$
p_t = \text{softmax}(\ell_t), \quad (\text{idx}_t, w_t) = \text{TopK}(p_t, k)
$$

$$
w \leftarrow \frac{w}{\sum w}
$$

### 4.2 Edge Generator (spider web)

$$
e_t = \text{LoopEmbed}(t)
$$

$$
g_t = \sigma\big([h_{t-1} \,;\, e_t]\, W_g\big)
$$

$$
A_t = \text{SparseTopK}(g_t, m)
$$

$$
\alpha_{ij} = \text{softmax over neighbors}
$$

### 4.3 Sparse Graph Message Passing

$$
m_t = \sum_{(i,j) \in A_t} \alpha_{ij} \cdot E_j(h_{t-1})
$$

> **Only active edges.** Dense multiply forbidden.

### 4.4 Expert FFN (Grouped GEMM)

$$
\Delta_t = \sum_{k=1}^{K} w_{t,k} \cdot \text{FFN}_{\text{idx}_{t,k}}(m_t)
$$

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x)
$$

### 4.5 Loop Epilogue (LOOM stability)

$$
u_t = h_{t-1} + \alpha \cdot \Delta_t + \beta \cdot x_0
$$

$$
h_t = \text{LayerNorm}(u_t)
$$

| Symbol | Meaning | Range |
|---|---|---|
| α | residual scale | 0.3 – 0.7 |
| β | re-inject weight | 0.05 – 0.2 |
| x₀ | original input embed | — |

### 4.6 Early Exit Gate (inference only)

$$
c_t = \sigma(h_t \cdot w_{\text{exit}})
$$

$$
\text{if } c_t > \tau \Rightarrow \text{exit at loop } t
$$

Threshold τ = 0.7 – 0.95.

### 4.7 Load Balance Loss (training)

$$
L_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

- `fᵢ` = fraction of tokens routed to expert `i`
- `pᵢ` = mean routing probability for expert `i`
- `N`  = num_experts

### 4.8 Total Training Loss

$$
L = L_{\text{CE}} + \lambda_1 L_{\text{aux}} + \lambda_2 L_{\text{edge}}
$$

- `L_CE`   = cross-entropy next-token
- `L_edge` = edge load-balance loss
- `λ₁, λ₂ ≈ 0.01`

---

## 5. Stability Rationale

| Problem | Fix (in formula) |
|---|---|
| Variance blow-up | `α · Δ` (bounded scale) |
| Representation drift | `β · x₀` (re-inject) |
| Weight-sharing symmetry | `bᵣ[t]` (per-loop bias) |
| Expert collapse | `Aₜ` (dynamic graph) |
| Edge bottleneck | `L_edge` (aux loss) |
| Compute waste | `cₜ > τ` (early exit) |

**Why each term exists:**

- **Residual scaling (`α`)** — prevents hidden-state variance from growing exponentially across loops
- **Input re-injection (`β`)** — keeps the model anchored to the original token representation
- **Per-loop bias (`bᵣ[t]`)** — breaks symmetry so each loop selects different experts
- **Dynamic edges (`Aₜ`)** — forces computational diversity across loops
- **Edge aux loss** — prevents a few edges from dominating
- **Early exit** — saves compute when confidence is high

---

## 6. Dimension Reference

| Symbol | Shape | Meaning |
|---|---|---|
| B | scalar | batch size |
| T | scalar | sequence length |
| H | scalar | hidden size |
| V | scalar | vocab size |
| E | scalar | num_experts |
| K | scalar | top_k (active per token) |
| m | scalar | edges_per_expert |
| r | scalar | num_loops |
| d | scalar | edge embed dim |
| hₜ | `[B, T, H]` | hidden state |
| ℓₜ, pₜ | `[B, T, E]` | router logits / probs |
| Aₜ (CSR) | ≤ `E·m` edges | sparse adjacency |
| Δₜ | `[B, T, H]` | expert output |
| cₜ | `[B, T]` | exit confidence |

---

## 7. Kernel ↔ Formula Map

| Kernel | Formula it computes |
|---|---|
| `router_fused` | ℓₜ, pₜ, TopK + loop bias |
| `graph_message_passing` | mₜ = Σ αᵢⱼ·Eⱼ(hₜ₋₁) |
| `expert_grouped_gemm` | Δₜ = Σ wₖ·FFN_k(mₜ) |
| `loop_epilogue_fused` | hₜ = LN(h + αΔ + βx₀) |
| `early_exit_gate` | cₜ = σ(hₜ·w_exit) > τ ? |

### Backend matrix

| Kernel | CUDA | ARM NEON/SVE | Scalar |
|---|---|---|---|
| `router_fused` | ✅ | ✅ | ✅ |
| `graph_message_passing` | ✅ | ✅ | ✅ |
| `expert_grouped_gemm` | ✅ | ✅ | ✅ |
| `loop_epilogue_fused` | ✅ | ✅ | ✅ |
| `early_exit_gate` | ✅ | ✅ | ✅ |

---

## 8. End-to-End Symbolic Flow

```
x₀ = Embed(tokens)

h₀ = Prelude(x₀)

for t in 1..=r:
    ℓₜ  = hₜ₋₁·Wᵣ + bᵣ[t]
    pₜ  = softmax(ℓₜ)
    (idx,w) = TopK(pₜ, k)

    eₜ  = LoopEmbed(t)
    gₜ  = σ([hₜ₋₁;eₜ]·W_g)
    Aₜ  = SparseTopK(gₜ, m)

    mₜ  = Σ αᵢⱼ·Eⱼ(hₜ₋₁)     over (i,j)∈Aₜ

    Δₜ  = Σ wₖ·FFN_{idxₖ}(mₜ)

    uₜ  = hₜ₋₁ + αΔₜ + βx₀
    hₜ  = LayerNorm(uₜ)

    if σ(hₜ·w_exit) > τ:  break   (inference only)

z = Coda(h_r)
logits = z·Wₒ
```

---

## 9. Default Config

| Parameter | Value | Meaning |
|---|---|---|
| hidden `H` | 512 | model width |
| num_experts `E` | 8 | MoE pool |
| top_k `K` | 2 | active/token |
| edges/expert `m` | 3 | web density |
| num_loops `r` | 6 | depth |
| residual `α` | 0.5 | stability |
| re-inject `β` | 0.1 | stability |
| exit `τ` | 0.85 | confidence |
| active params | ~150M | per-token |
| total params | ~400M | stored |

---

## 10. Full Stack

```
┌──────────────────────────────────────────────────────┐
│  CLI:  spider train | infer | bench                  │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│  RUST (Cargo workspace)                              │
│                                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│  │spider-core │ │spider-     │ │spider-     │        │
│  │model|graph │ │trainer     │ │infer       │        │
│  │router|stab │ │optimizer   │ │KV cache    │        │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘        │
│        └──────────────┼──────────────┘               │
│                       ▼                              │
│              ┌──────────────────┐                    │
│              │ spider-kernels   │  ← safe FFI        │
│              └────────┬─────────┘                    │
└───────────────────────┼──────────────────────────────┘
                        │ extern "C"
┌───────────────────────▼──────────────────────────────┐
│  C++ (CMake)                                         │
│              ┌──────────────────┐                    │
│              │  dispatch.cpp    │                    │
│              └────────┬─────────┘                    │
│        ┌──────────────┼──────────────┐               │
│        ▼              ▼              ▼               │
│   ┌────────┐    ┌──────────┐   ┌──────────┐         │
│   │ CUDA   │    │ ARM      │   │ Scalar   │         │
│   │sm_70+  │    │NEON/SVE  │   │ (ref)    │         │
│   └────────┘    └──────────┘   └──────────┘         │
└──────────────────────────────────────────────────────┘
```

---

## 11. Cheat Sheet

```
╔═══════════════════════════════════════════════════════════╗
║           SPIDER-WEB LOOPED MoE — CHEAT SHEET             ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   ARCH:    Prelude(2) → [SpiderMoE × r] → Coda(2)         ║
║   LOOP:    r = 5–9, shared params, per-loop bias          ║
║   MoE:     E = 8, k = 2, m = 3 edges/expert               ║
║                                                           ║
║   STABILITY:  hₜ = LN(h + αΔ + βx₀)                       ║
║               α = 0.5,  β = 0.1                           ║
║                                                           ║
║   EXIT:    cₜ = σ(hₜ·w_exit) > 0.85  (inference)          ║
║                                                           ║
║   LOSS:    L = L_CE + 0.01·L_aux + 0.01·L_edge            ║
║                                                           ║
║   KERNELS: router → graph → GEMM → epilogue → exit        ║
║   BACKENDS: CUDA | ARM NEON/SVE | Scalar                  ║
║                                                           ║
║   KEY INSIGHT:                                            ║
║     • Loop   → effective depth                            ║
║     • MoE    → parameter capacity                         ║
║     • Web    → computational diversity (anti-collapse)    ║
║     • α, β   → numerical stability                        ║
║     • exit   → inference speedup                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## References

1. **LOOM** — *Looping Meets MoE* (stability techniques)
2. **Foil** — *How to Loop MoE* (expert flattening, untied attention)
3. **GraphMoE** — Dynamic edge generation via virtual node
4. **Huginn** — Recurrent depth architecture
5. **MoE-Infinity** — Sparse MoE kernel design
6. **nano-vllm** — Inference engine structure

---

*Document version: 1.0 · Target: personal LLM (100M–1B)*