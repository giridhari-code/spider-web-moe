# ⚡ Spider-Web Looped MoE — Kernels

> Complete kernel specification for CUDA + ARM + Scalar backends.
> Each kernel: signature, memory layout, pseudocode, optimization notes.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Backend Matrix](#2-backend-matrix)
3. [Kernel 1 — router_fused](#3-kernel-1--router_fused)
4. [Kernel 2 — graph_message_passing](#4-kernel-2--graph_message_passing)
5. [Kernel 3 — expert_grouped_gemm](#5-kernel-3--expert_grouped_gemm)
6. [Kernel 4 — loop_epilogue_fused](#6-kernel-4--loop_epilogue_fused)
7. [Kernel 5 — early_exit_gate](#7-kernel-5--early_exit_gate)
8. [Memory Layouts](#8-memory-layouts)
9. [Numerical Precision](#9-numerical-precision)
10. [CUDA Backend Details](#10-cuda-backend-details)
11. [ARM Backend Details](#11-arm-backend-details)
12. [Scalar Reference](#12-scalar-reference)
13. [Testing Strategy](#13-testing-strategy)
14. [Performance Targets](#14-performance-targets)
15. [Optimization Checklist](#15-optimization-checklist)
16. [References](#16-references)

---

## 1. Overview

### 1.1 Design Principles

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  1. No host-device sync inside loops                     │
│     → CUDA Graphs compatible                             │
│                                                          │
│  2. FP32 accumulate everywhere critical                  │
│     → Router, epilogue, early exit                       │
│                                                          │
│  3. Fuse aggressively                                    │
│     → Memory round-trips are the enemy                   │
│                                                          │
│  4. Backend-agnostic signature                           │
│     → Same extern "C" for CUDA/ARM/Scalar                │
│                                                          │
│  5. Sparse stays sparse                                  │
│     → Never densify the graph                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Kernel Pipeline (one loop iteration)

```
   h_{t-1}                                              h_t
      │                                                  ▲
      ▼                                                  │
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐
  │  router  │──▶│  graph   │──▶│  expert  │──▶│  epilogue   │
  │  _fused  │   │ _message │   │ _grouped │   │   _fused    │
  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘
                                                      │
                                                      ▼
                                               ┌──────────┐
                                               │  early   │
                                               │  _exit   │
                                               └──────────┘
```

Har kernel independent hai, lekin **fused as a graph** chalta hai.

---

## 2. Backend Matrix

| Kernel | CUDA | ARM NEON/SVE | Scalar |
|---|---|---|---|
| `router_fused` | ✅ warp top-k | ✅ partial sort | ✅ bubble |
| `graph_message_passing` | ✅ atomic scatter | ✅ gather/scatter | ✅ loop |
| `expert_grouped_gemm` | ✅ mma.sync | ✅ outer product | ✅ dot |
| `loop_epilogue_fused` | ✅ vector ops | ✅ NEON FMA | ✅ scalar |
| `early_exit_gate` | ✅ warp reduce | ✅ reduction | ✅ dot |

**Runtime dispatch:** `dispatch.cpp` pehle backend detect karta hai, phir function pointer select karta hai.

---

## 3. Kernel 1 — `router_fused`

### 3.1 Purpose

Per-loop expert routing with **symmetry breaking** via `loop_idx`.

### 3.2 Signature

```c
void spider_router_fused(
    const void* logits,     // [B, E]   input (fp32/fp16/bf16)
    int dtype,              // 0=fp32, 1=fp16, 2=bf16
    int B,                  // batch size
    int E,                  // num_experts
    int K,                  // top_k
    int loop_idx,           // per-loop perturbation
    const void* loop_bias,  // [num_loops, E]  per-loop bias
    void* out_weights,      // [B, K]   normalized
    int*  out_indices,      // [B, K]   expert ids
    float* out_aux_loss,    // scalar, accumulated
    void* stream_handle
);
```

### 3.3 Memory Layout

```
logits:      row-major,  [B, E],  contiguous
loop_bias:   row-major,  [num_loops, E],  loop_idx-th row used
out_weights: row-major,  [B, K]
out_indices: row-major,  [B, K]

Alignment: 128 bytes
```

### 3.4 Pseudocode

```
for b in 0..B  (parallel):
    # 1. Add loop bias
    for e in 0..E:
        l[e] = logits[b, e] + loop_bias[loop_idx, e]

    # 2. Softmax (FP32 accumulate)
    max_l = max(l)
    sum_exp = 0
    for e in 0..E:
        p[e] = exp(l[e] - max_l)
        sum_exp += p[e]
    for e in 0..E:
        p[e] /= sum_exp

    # 3. Top-K selection
    (idx, w) = TopK(p, K)

    # 4. Normalize weights
    w_sum = sum(w)
    for k in 0..K:
        out_weights[b, k] = w[k] / w_sum
        out_indices[b, k] = idx[k]

    # 5. Load-balance loss (atomic add)
    for k in 0..K:
        atomic_add(out_aux_loss, p[idx[k]])
```

### 3.5 CUDA-Specific

```
Grid:   B blocks
Block:  128 threads (E ≤ 128 experts)
Shared: E floats (softmax buffer)

Optimization:
  • Warp-level top-k: each warp handles one row
  • __shfl_down for reduction
  • __ballot_sync for argmax
  • Loop bias loaded to registers (E ≤ 128)
  • aux_loss via atomicAdd (relaxed memory order)
```

### 3.6 ARM-Specific

```
Vectorize softmax with NEON:
  float32x4_t l = vld1q_f32(&logits[b*E]);
  ...
  
Top-K (K ≤ 4) via partial selection:
  • Find max iteratively (K times)
  • Or: sort 4-element blocks, merge
  
SVE variant:
  • Predicated loops over E
  • svsel_f32 for max
```

### 3.7 Performance Target

| Metric | Target |
|---|---|
| Latency | < 5% of loop time |
| Bandwidth | Read B·E·4 bytes, write B·K·8 bytes |
| Bottleneck | Softmax exp (compute) |

---

## 4. Kernel 2 — `graph_message_passing`

### 4.1 Purpose

Sparse spider-web traversal: token → expert A → edge → expert B.

### 4.2 Signature

```c
void spider_graph_message_passing(
    const void* hidden,         // [B, H]
    int dtype,
    int B, int H,
    const int*  adj_row_ptr,    // [E+1]  CSR offsets
    const int*  adj_col_idx,    // [nnz]  edge targets
    const void* edge_weights,   // [num_loops, nnz]
    int loop_idx,
    const int*  expert_assign,  // [B, K]  from router
    void* messages,             // [B, H]  output
    void* stream_handle
);
```

### 4.3 CSR Format

```
Example: E=4 experts, m=2 edges each

  row_ptr = [0, 2, 4, 6, 8]
  col_idx = [2, 3, 0, 2, 1, 3, 0, 1]
  
  Expert 0 → {2, 3}
  Expert 1 → {0, 2}
  Expert 2 → {1, 3}
  Expert 3 → {0, 1}

nnz = row_ptr[E] = 8
```

### 4.4 Memory Layout

```
Per-loop edge weights:
  edge_weights[loop_idx] → [nnz] float

Total edge storage:
  num_loops × nnz × 4 bytes
  = 6 × 8 × 4 = 192 bytes  (negligible)
```

### 4.5 Pseudocode

```
zero(messages)  # [B, H]

for b in 0..B  (parallel):
    for k in 0..K:
        src = expert_assign[b, k]
        if src == INACTIVE: continue

        # Gather source hidden
        h_src = hidden[b, :]   # [H]

        # Walk edges
        for e in row_ptr[src]..row_ptr[src+1]:
            dst  = col_idx[e]
            w    = edge_weights[loop_idx, e]

            # Scatter to dst (atomic add)
            for i in 0..H:
                atomic_add(messages[b, i], w * h_src[i])
```

### 4.6 CUDA-Specific

```
Grid:   (B, H/128) — blocks over batch and hidden tiles
Block:  128 threads

Optimization:
  • Atomic add → shared-mem reduction first, then global atomic
  • Coalesce hidden reads: consecutive H reads
  • Edge weights cached in shared mem (nnz small)
  • Warp-level scatter when H fits in register file

Avoid:
  • Dense adjacency multiply — catastrophic
  • Divergent atomic paths — serialize

Better pattern:
  1. Load h_src to shared mem [H]
  2. Warp-cooperative: each thread handles H/32 elements
  3. Scatter to all dst experts with non-atomic (since dst rare)
```

### 4.7 ARM-Specific

```
Single-threaded per batch row:
  for b in batch:
    zero local buffer[H]
    for k, edges:
      NEON load h_src[0..H/4]
      NEON mul by weight
      NEON add to local buffer
    copy local buffer to messages[b]

Threading:
  • Batch parallelized via rayon
  • Per-expert parallelism possible if E > threads
```

### 4.8 Performance Target

| Metric | Target |
|---|---|
| Latency | < 15% of loop time |
| Bandwidth | Read B·H·4, write B·H·4, read nnz·4 |
| Bottleneck | Atomics (CUDA), memory (ARM) |

---

## 5. Kernel 3 — `expert_grouped_gemm`

### 5.1 Purpose

MoE expert FFN — grouped GEMM with permute/unpermute.

### 5.2 Signature

```c
void spider_expert_grouped_gemm(
    const void* tokens,         // [B, H]  expert inputs
    int dtype,
    int B, int H, int H_up,     // H_up = 4·H (FFN expansion)
    int E, int K,
    const int*  expert_assign,  // [B, K]
    const int*  token_perm,     // [B·K]   sorted by expert
    const int*  perm_offsets,   // [E]     start offset per expert
    const void* W_up,           // [E, H, H_up]
    const void* W_down,         // [E, H_up, H]
    void* output,               // [B, H]
    void* stream_handle
);
```

### 5.3 Grouped GEMM Structure

```
             ┌─────────┐
   Tokens    │ Permute │    [B·K, H]
   [B,H]     └────┬────┘
                  │  grouped by expert
                  ▼
        ┌─────────────────────┐
        │  Batched GEMM       │
        │  Expert_i: [Ni, H]  │
        │    × W_up[i]        │  → [Ni, H_up]
        │    × W_down[i]      │  → [Ni, H]
        └────────┬────────────┘
                 │  per-expert outputs
                 ▼
        ┌─────────────────────┐
        │  Unpermute + combine│  weight by routing
        │  → [B, H]           │
        └─────────────────────┘
```

### 5.4 Pseudocode

```
# Phase 1: Permute
for b in 0..B:
    for k in 0..K:
        e = expert_assign[b, k]
        slot = perm_offsets[e] + slot_counter[e]
        permuted_tokens[slot] = tokens[b]
        permuted_weights[slot] = routing_weight[b, k]
        slot_counter[e]++

# Phase 2: Grouped GEMM (per expert)
for e in 0..E:
    start = perm_offsets[e]
    end   = perm_offsets[e+1]
    N_e   = end - start

    if N_e == 0: continue

    # Up projection
    tmp = permuted_tokens[start:end] @ W_up[e]     # [N_e, H_up]
    tmp = gelu(tmp)                                 # activation

    # Down projection
    out_e = tmp @ W_down[e]                        # [N_e, H]

    permuted_output[start:end] = out_e

# Phase 3: Unpermute + combine
for b in 0..B:
    for k in 0..K:
        slot = find_slot(b, k)
        output[b] += permuted_weights[slot] * permuted_output[slot]
```

### 5.5 CUDA-Specific

```
Approach: Split-K with expert-parallel blocks

Grid:   (num_tiles_H_up, num_experts) 
Block:  (128, K=16 tiles)

Optimization:
  • mma.sync for matrix tiles
  • cp.async for W_up loading
  • Padding-free: handle N_e < tile
  • Shared mem: token_tile [128, 64] + W_tile [64, 64]

Avoid:
  • Padded batched GEMM (wasted flops)
  • Dynamic N_e → use perm_offsets to compute tile count

Fallback:
  • cuBLAS grouped GEMM (baseline, but slow for small N_e)
  • Triton grouped_gemm (custom)
```

### 5.6 ARM-Specific

```
Small expert counts (E ≤ 16):
  • K-blocked GEMM
  • NEON: vfmaq_f32 accumulation
  • Cache blocking: H=512 → 128 tiles

Strategy:
  • Single-pass GEMM per expert
  • L1 cache W tile, L2 cache tokens
  • Thread pool: rayon par_iter over experts
```

### 5.7 Performance Target

| Metric | Target |
|---|---|
| FLOPs utilization | > 60% peak (CUDA), > 40% (ARM) |
| Latency | > 40% of loop time (dominant) |
| Bottleneck | HBM bandwidth for W matrices |

---

## 6. Kernel 4 — `loop_epilogue_fused`

### 6.1 Purpose

Loop end fusion: residual + re-inject + LayerNorm in **one pass**.

### 6.2 Signature

```c
void spider_loop_epilogue_fused(
    const void* h,              // [B, H]  current hidden
    const void* delta,          // [B, H]  expert output
    const void* input_embed,    // [B, H]  original x₀
    int dtype,
    int B, int H,
    float alpha,                // 0.5
    float beta,                 // 0.1
    const void* ln_weight,      // [H]
    const void* ln_bias,        // [H]
    void* h_next,               // [B, H]
    void* stream_handle
);
```

### 6.3 Formula

$$
u = h + \alpha \cdot \Delta + \beta \cdot x_0
$$

$$
h_{\text{next}} = \text{LayerNorm}(u) = \frac{u - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta_{\text{ln}}
$$

### 6.4 Two-Pass vs One-Pass

```
Two-pass (naive):
  Pass 1: compute mean, variance      (read u once)
  Pass 2: normalize                   (read u again)
  → 2× memory traffic

One-pass (Welford):
  Simultaneously compute mean & var while accumulating
  → 1× memory traffic  ✅
```

**Mandatory:** one-pass Welford for memory-bound kernel.

### 6.5 Pseudocode (One-Pass)

```
for b in 0..B  (parallel):
    # Phase 1: compute u, track Welford stats
    mean = 0
    M2   = 0
    for i in 0..H:
        u[i] = h[b,i] + alpha*delta[b,i] + beta*input_embed[b,i]
        
        # Welford update
        delta_w = u[i] - mean
        mean   += delta_w / (i + 1)
        M2     += delta_w * (u[i] - mean)
    
    var = M2 / H
    inv_std = 1 / sqrt(var + eps)

    # Phase 2: normalize
    for i in 0..H:
        h_next[b,i] = (u[i] - mean) * inv_std * ln_weight[i] + ln_bias[i]
```

### 6.6 CUDA-Specific

```
Grid:   B blocks
Block:  256 threads (H/256 elements per thread)

Optimization:
  • Vectorized loads: float4
  • Warp-level reduction for mean/var (Welford)
  • Shared mem for block reduction (256 → 8 warps → 1)
  • Write normalized in same block (u in registers)

Register pressure:
  • H=512, 256 threads → 2 floats/thread
  • u values kept in registers
```

### 6.7 ARM-Specific

```
NEON Welford:
  float32x4_t u = vmlaq_f32(h, alpha_v, delta);
  u = vmlaq_f32(u, beta_v, input_embed);
  
  # Welford (vectorized):
  float32x4_t delta = vsubq_f32(u, mean_v);
  mean_v = vaddq_f32(mean_v, vdivq_f32(delta, n_v));
  M2_v   = vmlaq_f32(M2_v, delta, vsubq_f32(u, mean_v));

  # Horizontal reduce after loop

SVE:
  • Predicated loop over H
  • svmla_f32 for FMA chains
  • svreduce for mean/var
```

### 6.8 Performance Target

| Metric | Target |
|---|---|
| Latency | < 10% of loop time |
| Bandwidth | Read 3·B·H·4, write B·H·4 |
| Bottleneck | Memory bandwidth (100% HBM-bound) |

**Best case:** Achieve 80% of peak HBM bandwidth.

---

## 7. Kernel 5 — `early_exit_gate`

### 7.1 Purpose

Per-token confidence check at loop boundary (inference only).

### 7.2 Signature

```c
void spider_early_exit_gate(
    const void* h,              // [B, H]
    const void* exit_proj,      // [H]     learned
    int dtype,
    int B, int H,
    float threshold,            // 0.85
    bool*  should_exit,         // [B]     per-token mask
    void* stream_handle
);
```

### 7.3 Formula

$$
c_i = \sigma\left(\sum_{j=1}^{H} h_{i,j} \cdot w_j\right)
$$

$$
\text{exit}_i = (c_i > \tau)
$$

### 7.4 Pseudocode

```
for b in 0..B  (parallel):
    dot = 0
    for j in 0..H:
        dot += h[b, j] * exit_proj[j]
    
    conf = sigmoid(dot)
    should_exit[b] = (conf > threshold)
```

### 7.5 CUDA-Specific

```
Grid:   B blocks
Block:  128 threads (H/128 per thread)

Optimization:
  • Warp reduction for dot product
  • __shfl_down_sync
  • exit_proj cached in shared mem (H ≤ 1024)
  • No shared mem if H fits in registers
```

### 7.6 ARM-Specific

```
NEON:
  float32x4_t acc = vdupq_n_f32(0);
  for (i = 0; i < H; i += 4) {
      acc = vfmaq_f32(acc, h_vec, proj_vec);
  }
  dot = vaddvq_f32(acc);

Threading:
  • Batch parallel via rayon
  • Small kernel — launch overhead dominates
```

### 7.7 Performance Target

| Metric | Target |
|---|---|
| Latency | < 2% of loop time |
| Bandwidth | Read B·H·4, write B |
| Bottleneck | Launch overhead for small B |

---

## 8. Memory Layouts

### 8.1 Tensor Alignment

```
All tensors: 128-byte aligned

┌────────────────────────────────────┐
│  h tensor [B, T, H]                │
│                                    │
│  b=0,t=0: [H floats]   ─ 128-aligned│
│  b=0,t=1: [H floats]   ─ 128-aligned│
│  ...                                │
│                                    │
│  H=512 → 2KB per row               │
│  B·T=2048 → 4MB total              │
└────────────────────────────────────┘
```

### 8.2 Sparse Graph (CSR)

```
┌────────────────────────────────────┐
│  row_ptr:   [E+1]     int32        │
│  col_idx:   [nnz]     int32        │
│  weights:   [r, nnz]  fp32         │
└────────────────────────────────────┘

Persistent between loops.
Only `loop_idx` selects weight slice.
```

### 8.3 Expert Weights

```
┌────────────────────────────────────┐
│  W_up:   [E, H, H_up]   fp16       │
│  W_down: [E, H_up, H]   fp16       │
│                                    │
│  Contiguous per expert — cache      │
│  friendly for grouped GEMM.        │
└────────────────────────────────────┘
```

---

## 9. Numerical Precision

### 9.1 Precision Rules

```
┌──────────────────────────┬─────────────┬─────────────┐
│  Operation               │  Compute    │  Storage    │
├──────────────────────────┼─────────────┼─────────────┤
│  Router softmax          │  FP32       │  FP16/BF16  │
│  Top-K indices           │  int32      │  int32      │
│  Graph message           │  FP32       │  FP16/BF16  │
│  Expert GEMM             │  FP32 accum │  FP16/BF16  │
│  Epilogue Welford        │  FP32       │  FP16/BF16  │
│  Early exit dot          │  FP32       │  FP32       │
└──────────────────────────┴─────────────┴─────────────┘
```

### 9.2 Why FP32 Accumulate?

```
FP16 accumulate error:
  • 1024-element dot product → up to 1e-2 abs error
  • Loops amplify → divergence

FP32 accumulate:
  • Same dot → 1e-5 abs error
  • Safe across r=9 loops

Rule: any reduction → FP32.
```

### 9.3 Tolerance Table

| Test | Tolerance |
|---|---|
| Scalar vs Scalar | exact |
| CUDA vs Scalar (FP32) | 1e-4 |
| ARM vs Scalar (FP32) | 1e-4 |
| FP16 vs FP32 ref | 1e-2 |
| BF16 vs FP32 ref | 5e-2 |

---

## 10. CUDA Backend Details

### 10.1 Target Architectures

| GPU | Arch | Features |
|---|---|---|
| V100 | sm_70 | Warp primitives |
| A100 | sm_80 | cp.async, mma.sync |
| RTX 4090 | sm_89 | FP8, wgmma (partial) |
| H100 | sm_90 | Full wgmma, TMA |

### 10.2 Stream Strategy

```
Loop iteration t:
   Stream_i (round-robin per iteration):
     router → graph → GEMM → epilogue → exit

CUDA Graphs:
  Loop body captured once,
  launched per iteration with loop_idx change.
```

### 10.3 Kernel Launch Configuration

```
router_fused:
  Grid:   B
  Block:  128
  Smem:   E * 4 bytes

graph_message_passing:
  Grid:   (B, H/128)
  Block:  128
  Smem:   H * 4 bytes (h_src)

expert_grouped_gemm:
  Grid:   (H_up/128, E)
  Block:  (128, 16)
  Smem:   (128*64 + 64*64) * 4 bytes

loop_epilogue_fused:
  Grid:   B
  Block:  256
  Smem:   256 * 8 bytes (Welford reduction)

early_exit_gate:
  Grid:   B
  Block:  128
  Smem:   H * 4 bytes (optional)
```

### 10.4 CUDA Graphs

```
Critical for:
  • Loop iteration (r=6 → 6 launches of 5 kernels = 30 launches)
  • Reduce to 1 graph launch

Setup:
  graph = cudaGraphCreate()
  for t in 0..r:
      capture_loop_iteration(graph, t)
  cudaGraphInstantiate(graph)
  cudaGraphLaunch(graph, stream)
```

---

## 11. ARM Backend Details

### 11.1 Target Platforms

| Platform | Features |
|---|---|
| Apple M1/M2/M3 | NEON, AMX (limited) |
| AWS Graviton 3/4 | NEON + SVE/SVE2 |
| Ampere Altra | NEON |

### 11.2 Threading Model

```
Rust rayon:
  par_iter over batch rows for small kernels
  par_iter over experts for grouped GEMM

Thread pool: 8-16 threads (cores)
  Batch = 32 → 4 rows/thread

Avoid:
  • Over-subscription (thread count > cores)
  • Sync between kernels (let them run async in rayon)
```

### 11.3 Cache Blocking

```
L1 = 64 KB
L2 = 1 MB (M1) or 2 MB (Graviton)

For epilogue (H=512):
  h, delta, input_embed, h_next:
    4 × 2 KB = 8 KB  → fits L1

For grouped GEMM (H=1024):
  W_up = 4 MB → L2 bottleneck
  Strategy: block H into 256-tiles
```

### 11.4 SVE Predication

```
SVE for Graviton:
  svfloat32_t acc = svdup_f32(0);
  for (i = 0; i < H; i += svcntw()) {
      svbool_t pg = svwhilelt_b32(i, H);
      svfloat32_t h = svld1(pg, h_arr + i);
      svfloat32_t w = svld1(pg, w_arr + i);
      acc = svmla_f32_x(pg, acc, h, w);
  }
  dot = svaddv_f32(svptrue_b32(), acc);
```

---

## 12. Scalar Reference

### 12.1 Purpose

Pure C++ reference for **correctness verification only**.

### 12.2 Structure

```cpp
// router_scalar.cpp
void spider_router_scalar(...) {
    for (int b = 0; b < B; ++b) {
        std::vector<float> l(E);
        for (int e = 0; e < E; ++e)
            l[e] = logits[b*E + e] + loop_bias[loop_idx*E + e];
        
        float maxl = *std::max_element(l.begin(), l.end());
        float sum = 0;
        for (auto& x : l) { x = std::exp(x - maxl); sum += x; }
        for (auto& x : l) x /= sum;
        
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
            [&](int a, int b) { return l[a] > l[b]; });
        
        float wsum = 0;
        for (int k = 0; k < K; ++k) wsum += l[idx[k]];
        for (int k = 0; k < K; ++k) {
            out_weights[b*K + k] = l[idx[k]] / wsum;
            out_indices[b*K + k] = idx[k];
        }
    }
}
```

### 12.3 Rules

- **No SIMD**, no threading
- **FP64 where feasible** (extra reference)
- **Never used in production** — only tests

---

## 13. Testing Strategy

### 13.1 Parity Tests

```
┌─────────────────────────────────────────────────┐
│  For each kernel:                               │
│                                                 │
│  1. Generate random inputs (FP32, fixed seed)  │
│  2. Run scalar reference                        │
│  3. Run CUDA version                            │
│  4. Run ARM version                             │
│  5. Compare all three within tolerance          │
│                                                 │
│  Repeat for FP16, BF16                          │
└─────────────────────────────────────────────────┘
```

### 13.2 Edge Cases

```
router_fused:
  • B=1 (single token)
  • E=1 (degenerate MoE)
  • K=E (all experts active)

graph_message_passing:
  • nnz=0 (empty graph)
  • Full graph (nnz = E·(E-1))
  • Self-loops (i → i)

expert_grouped_gemm:
  • Expert with N_e=0 (unused expert)
  • All tokens to one expert (pathological)
  • N_e much smaller than tile

loop_epilogue_fused:
  • H not multiple of SIMD width
  • All-zero input (Welford numerical)

early_exit_gate:
  • conf exactly = threshold
  • All tokens exit at loop 1
```

### 13.3 Benchmarks

```
Per kernel:
  • Latency (median of 100 runs)
  • Bandwidth (GB/s achieved)
  • FLOPs (for GEMM)

End-to-end:
  • Tokens/sec (train)
  • Tokens/sec (infer, with early exit)
  • VRAM usage
  • Peak FLOPs utilization
```

---

## 14. Performance Targets

### 14.1 Latency Budget (per loop iteration, H=512, B=32)

```
┌──────────────────────────┬─────────┬──────────┐
│  Kernel                  │  Target │  % loop  │
├──────────────────────────┼─────────┼──────────┤
│  router_fused            │  0.05ms │    5%    │
│  graph_message_passing   │  0.15ms │   15%    │
│  expert_grouped_gemm     │  0.40ms │   40%    │
│  loop_epilogue_fused     │  0.10ms │   10%    │
│  early_exit_gate         │  0.02ms │    2%    │
├──────────────────────────┼─────────┼──────────┤
│  Total per loop          │  0.72ms │   —      │
│  × 6 loops               │  4.3ms  │          │
└──────────────────────────┴─────────┴──────────┘
```

### 14.2 End-to-End Targets

| Metric | Target |
|---|---|
| Train tok/s (RTX 4090) | > 8,000 |
| Infer tok/s (RTX 4090) | > 30,000 (with exit) |
| VRAM (train, B=32) | < 8 GB |
| Speedup vs eager PyTorch | 2–3× |

---

## 15. Optimization Checklist

### 15.1 Must-Have

- [ ] FP32 accumulate for router/epilogue/exit
- [ ] CUDA Graphs for loop unrolling
- [ ] Fused epilogue (no memory round-trip)
- [ ] Sparse CSR for graph (never dense)
- [ ] Grouped GEMM without padding
- [ ] 128-byte alignment everywhere

### 15.2 Should-Have

- [ ] Vectorized loads (float4 in CUDA, NEON in ARM)
- [ ] Warp-level reductions in CUDA
- [ ] Welford one-pass for LayerNorm
- [ ] Per-loop stream overlap (CUDA)
- [ ] rayon parallelism (ARM)

### 15.3 Nice-to-Have

- [ ] FP8 expert GEMM (H100, RTX 4090)
- [ ] TMA async load (sm_90)
- [ ] SVE predicated loops (Graviton)
- [ ] Persistent kernels (loop stays resident)
- [ ] Speculative decode (future)

---

## 16. References

1. **FlashAttention-2** — Efficient attention patterns
2. **Megablocks** — Grouped GEMM for MoE
3. **MoE-Infinity** — Sparse MoE kernels
4. **Triton** — Fusion DSL
5. **CUTLASS** — CUDA GEMM templates
6. **ARM NEON** — Intrinsics reference
7. **Welford** — One-pass variance algorithm

---

## Appendix A: Kernel File Map

```
kernels/
├── include/spider_moe/
│   ├── router.h                    → Section 3
│   ├── graph_message_passing.h     → Section 4
│   ├── expert_grouped_gemm.h       → Section 5
│   ├── loop_epilogue.h             → Section 6
│   └── early_exit.h                → Section 7
│
├── src/                            (dispatch layer)
│   ├── dispatch.cpp
│   ├── router.cpp
│   ├── graph_message_passing.cpp
│   ├── expert_grouped_gemm.cpp
│   ├── loop_epilogue.cpp
│   ├── early_exit.cpp
│   └── bindings.cpp
│
├── cuda/
│   ├── router_fused.cu             → 3.5
│   ├── graph_message_passing.cu    → 4.6
│   ├── expert_grouped_gemm.cu      → 5.5
│   ├── loop_epilogue_fused.cu      → 6.6
│   └── early_exit_gate.cu          → 7.5
│
├── arm/
│   ├── router_neon.cpp             → 3.6
│   ├── graph_neon.cpp              → 4.7
│   ├── expert_gemm_neon.cpp        → 5.6
│   ├── epilogue_neon.cpp           → 6.7
│   └── early_exit_neon.cpp         → 7.6
│
├── scalar/
│   ├── router_scalar.cpp           → 12.2
│   ├── graph_scalar.cpp
│   ├── expert_gemm_scalar.cpp
│   ├── epilogue_scalar.cpp
│   └── early_exit_scalar.cpp
│
└── tests/
    ├── parity_test.cpp             → 13.1
    └── bench.cpp                   → 13.3
```

---

## Appendix B: Runtime Dispatch

```cpp
// dispatch.cpp
enum Backend { CUDA, ARM, SCALAR };

Backend detect_backend() {
#ifdef __CUDACC__
    int cuda_devices;
    if (cudaGetDeviceCount(&cuda_devices) == cudaSuccess && cuda_devices > 0)
        return CUDA;
#endif
#if defined(__ARM_NEON) || defined(__ARM_FEATURE_SVE)
    return ARM;
#endif
    return SCALAR;
}

// Function pointers per kernel
using router_fn = decltype(&spider_router_scalar);
router_fn g_router = nullptr;

void spider_init() {
    switch (detect_backend()) {
        case CUDA:   g_router = spider_router_cuda; break;
        case ARM:    g_router = spider_router_neon; break;
        case SCALAR: g_router = spider_router_scalar; break;
    }
}
```

---

*Document version: 1.0 · Target: personal LLM (100M–1B)*
*Status: Spec-complete · CUDA + ARM + Scalar backends*