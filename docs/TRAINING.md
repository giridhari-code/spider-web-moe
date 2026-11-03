# 🎓 Spider-Web Looped MoE — Training

> Complete training specification: pipeline, hyperparameters, stability monitoring, recipes.
> Target: personal-scale (100M–1B params) on single GPU.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Training Pipeline](#2-training-pipeline)
3. [Data Pipeline](#3-data-pipeline)
4. [Hyperparameters](#4-hyperparameters)
5. [Loss Composition](#5-loss-composition)
6. [Gradient Flow Through Loops](#6-gradient-flow-through-loops)
7. [Stability Monitoring](#7-stability-monitoring)
8. [Memory Optimization](#8-memory-optimization)
9. [Checkpointing](#9-checkpointing)
10. [Logging & Metrics](#10-logging--metrics)
11. [Recipes](#11-recipes)
12. [Troubleshooting](#12-troubleshooting)
13. [Reproducibility](#13-reproducibility)
14. [References](#14-references)

---

## 1. Overview

### 1.1 Training Phases

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Phase 1:  PRETRAIN                                          │
│  ─────────────────────                                       │
│  • Next-token prediction on raw corpus                       │
│  • Full r loops (no early exit)                              │
│  • Loss: CE + aux + edge                                     │
│  • Bulk of compute                                           │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 2:  SFT (Supervised Fine-Tuning)                      │
│  ─────────────────────────────────────                       │
│  • Instruction-following data                                │
│  • Same loss, instruction format                             │
│  • Fewer steps, lower lr                                     │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 3:  DPO (optional)                                    │
│  ────────────────────────                                    │
│  • Preference pairs                                          │
│  • Direct Preference Optimization                            │
│  • Alignment                                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Training Budget (personal scale)

| Phase | Tokens | Steps | Hours (RTX 4090) |
|---|---|---|---|
| Pretrain | 10B | 150K | ~72h |
| SFT | 100M | 3K | ~4h |
| DPO | 10M | 500 | ~1h |

**Total: ~77 hours on single 4090.**

---

## 2. Training Pipeline

### 2.1 High-Level Flow

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Data Loader                                             │
│    ↓  (mmap binary tokens, shuffle buffer 10K)           │
│                                                          │
│  Batch [B, T]                                            │
│    ↓                                                     │
│                                                          │
│  Forward:  full r loops                                  │
│    ↓  track per-loop stats                               │
│                                                          │
│  Loss = CE + λ₁·L_aux + λ₂·L_edge                        │
│    ↓                                                     │
│                                                          │
│  Backward:  BPTT through all loops                       │
│    ↓  gradient checkpointing                             │
│                                                          │
│  Optimizer:  AdamW + grad clip                           │
│    ↓                                                     │
│                                                          │
│  Checkpoint (every 1000 steps)                           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 One Training Step

```
┌─────────────────────────────────────────────────────────┐
│  STEP:                                                  │
│                                                         │
│  1. Load batch          [B=32, T=1024]                  │
│                                                         │
│  2. Forward:                                            │
│     h₀ = Prelude(embed(x))                              │
│     for t in 1..=r:                                     │
│         hₜ, statsₜ = SpiderMoE(hₜ₋₁, x₀, t)            │
│         track: var(hₜ), expert_load[t], edge_load[t]    │
│     z = Coda(h_r)                                       │
│     logits = z·Wₒ                                       │
│                                                         │
│  3. Loss:                                               │
│     L_CE   = cross_entropy(logits, targets)             │
│     L_aux  = load_balance_loss(expert_load)             │
│     L_edge = edge_balance_loss(edge_load)               │
│     L = L_CE + 0.01·L_aux + 0.01·L_edge                │
│                                                         │
│  4. Backward (BPTT):                                    │
│     checkpoint every loop boundary                      │
│     recompute forward when backward reaches loop t      │
│                                                         │
│  5. Optimizer step:                                     │
│     clip grad norm → 1.0                                │
│     AdamW step                                          │
│     lr cosine decay                                     │
│                                                         │
│  6. Log:  loss, lr, grad_norm, expert_load, stats       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.3 BPTT (Backprop Through Time)

```
Forward:                      Backward:
  loop 1                        loss
  loop 2                          │
  loop 3                          ▼
  loop 4                       loop r
  loop 5                        loop r-1
  loop 6                        loop r-2
                                  │
  (activation memory            loop 1
   grows linearly              
   with r)                      ∂L/∂θ

Standard BPTT: memory O(r).

With checkpointing:
  • Save only loop boundaries (r+1 checkpoints)
  • Recompute each loop's internals during backward
  • Memory O(r) → O(1) at 30% speed cost
```

**Mandatory:** gradient checkpointing for r ≥ 6.

---

## 3. Data Pipeline

### 3.1 Tokenized Data Format

```
tokens/
├── train/
│   ├── shard_000.bin    # uint32 token IDs
│   ├── shard_001.bin
│   └── ...
└── val/
    └── shard_000.bin

Format:
  • Pure binary
  • uint32 per token (or uint16 if vocab ≤ 65536)
  • No separators — model learns from <EOS>
  • Each shard ~1GB
```

### 3.2 Data Loader

```
┌────────────────────────────────────────────────────┐
│  Memory-mapped loader                              │
│                                                    │
│  1. mmap shard file (lazy load)                    │
│  2. Random offset sampling                         │
│  3. Extract sequence [T+1] tokens                  │
│     • Input:  tokens[0..T]                         │
│     • Target: tokens[1..T+1]                       │
│  4. Shuffle buffer: 10,000 samples                 │
│  5. Batch: 32 sequences                            │
│                                                    │
│  Total memory: ~batch_size × T × 4 bytes          │
│    = 32 × 1024 × 4 = 128 KB per batch             │
└────────────────────────────────────────────────────┘
```

### 3.3 Data Mixing (pretrain)

```
Recommended mix:
  • 60% web text (Common Crawl, filtered)
  • 20% code (GitHub, permissive)
  • 10% math (arxiv, textbooks)
  • 10% books / long-form

For personal use:
  • Start with single domain (e.g., code only)
  • Easier to debug
  • Lower entropy → faster convergence
```

### 3.4 SFT Data Format

```
Example:
{
  "instruction": "Write a Rust function to reverse a string.",
  "input": "",
  "output": "fn reverse(s: &str) -> String {\n    s.chars().rev().collect()\n}"
}

Tokenized with chat template:
  <|system|>You are a helpful assistant.<|end|>
  <|user|>Write a Rust function...<|end|>
  <|assistant|>fn reverse...<|end|>

Loss: only on assistant tokens (mask others).
```

---

## 4. Hyperparameters

### 4.1 Pretrain (default)

```yaml
# configs/train-pretrain.yaml
model:
  hidden_size: 512
  num_experts: 8
  top_k: 2
  edges_per_expert: 3
  num_loops: 6
  residual_scale: 0.5
  re_inject_weight: 0.1

training:
  batch_size: 32
  seq_len: 1024
  steps: 150_000
  grad_accum: 4            # effective batch = 128

  optimizer: adamw
  lr: 3.0e-4
  lr_schedule: cosine
  warmup_steps: 2000
  min_lr_ratio: 0.1
  weight_decay: 0.1
  grad_clip: 1.0

  loss_weights:
    ce: 1.0
    aux: 0.01
    edge: 0.01

  precision: bf16
  grad_checkpoint: true

logging:
  log_every: 10
  eval_every: 1000
  save_every: 5000
```

### 4.2 SFT

```yaml
# configs/train-sft.yaml
model:
  # same as pretrain
training:
  batch_size: 16
  seq_len: 2048
  steps: 3_000
  grad_accum: 2

  optimizer: adamw
  lr: 5.0e-5           # 6× lower than pretrain
  lr_schedule: cosine
  warmup_steps: 100
  weight_decay: 0.05

  loss_weights:
    ce: 1.0            # only on assistant tokens
    aux: 0.005
    edge: 0.005
```

### 4.3 Batch Size Scaling

```
Rule of thumb:
  lr ∝ sqrt(batch_size)

Baseline:
  batch = 32 → lr = 3e-4

Scaled:
  batch = 64 → lr = 3e-4 × √2 = 4.2e-4
  batch = 16 → lr = 3e-4 × 1/√2 = 2.1e-4
```

### 4.4 Loop Count Effect

```
r = 3:  fast, less depth, lower quality
r = 6:  ← sweet spot for personal scale
r = 9:  slow, marginal gain, memory pressure
r = 12: unstable, not recommended first pass

Recommendation:  start r=4, scale to 6 after 10K steps.
```

### 4.5 Learning Rate Schedule

```
lr
 │
 │  ╱────╲
 │ ╱      ╲
 │╱        ╲──────────
 │           cosine
 └───────────────────────► step
   warmup    plateau    decay

Warmup:   2000 steps → linear from 0 to lr
Plateau:  none — cosine continues
Decay:    cosine to lr × 0.1 by end
```

---

## 5. Loss Composition

### 5.1 Total Loss

$$
L = L_{\text{CE}} + \lambda_1 L_{\text{aux}} + \lambda_2 L_{\text{edge}}
$$

| Term | Weight | Purpose |
|---|---|---|
| `L_CE` | 1.0 | Main task (next-token) |
| `L_aux` | 0.01 | Expert load balance |
| `L_edge` | 0.01 | Edge load balance |

### 5.2 Load Balance Loss

$$
L_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

- `fᵢ` = fraction of tokens routed to expert i
- `pᵢ` = mean routing probability for expert i
- `N`  = num_experts

**Why:** Without it, router collapses to 1–2 experts.

**Target:** `L_aux ≈ 1.0` at convergence (perfectly balanced).

### 5.3 Edge Balance Loss

$$
L_{\text{edge}} = M \cdot \sum_{j=1}^{M} g_j \cdot q_j
$$

- `gⱼ` = fraction of messages through edge j
- `qⱼ` = mean edge weight
- `M`  = total edges (E × m)

**Why:** Prevents one edge from dominating the web.

**Target:** `L_edge ≈ 1.0` at convergence.

### 5.4 Loss Monitoring

```
Expected trajectory:

  Step        L_CE      L_aux     L_edge
  ─────────────────────────────────────────
  0           10.4      8.0       8.0
  1K          7.2       2.5       2.8
  10K         4.8       1.4       1.6
  50K         3.5       1.1       1.2
  150K        2.9       1.0       1.0

Warning signs:
  L_CE plateaus > 5.0 by 10K     → lr too low or data issue
  L_aux > 3.0 after 10K          → router collapse, increase λ₁
  L_edge > 3.0 after 10K         → edge collapse, increase λ₂
  L_CE suddenly jumps            → instability, check variance
```

---

## 6. Gradient Flow Through Loops

### 6.1 The Challenge

```
Forward:
  h₀ → [Loop₁] → h₁ → [Loop₂] → h₂ → ... → h_r → z → loss

Backward (chain rule):
  ∂L/∂θ = Σₜ ∂L/∂hₜ · ∂hₜ/∂θ

Each loop contributes to gradient.
Same θ used r times → ∂L/∂θ is sum of r contributions.
```

### 6.2 Vanishing/Exploding Gradients

```
Without stability:
  ∂hₜ/∂hₜ₋₁ ≈ 1.2 → after 6 loops: 1.2⁶ ≈ 3×
  after 9 loops: 1.2⁹ ≈ 5×

With α=0.5 residual scale:
  ∂hₜ/∂hₜ₋₁ ≈ 1 + 0.5·small → bounded near 1
  gradient stable across loops
```

### 6.3 Gradient Checkpointing

```
Memory comparison (H=512, B=32, T=1024, r=6):

Without checkpointing:
  activation storage = r × (per-loop activations)
  = 6 × 512 MB = 3 GB

With checkpointing:
  only loop boundaries stored = 7 × activation_size
  = 7 × 8 MB = 56 MB
  + recompute cost: ~30% slower

NET: 55× memory reduction, 30% speed cost.
```

### 6.4 Gradient Monitoring

```
Track per-loop gradient norm:
  ‖∇h_t‖  for t = 1..r

Expected pattern:
  ‖∇h_1‖ < ‖∇h_2‖ < ... < ‖∇h_r‖

Meaning: later loops closer to loss → larger grad.
Normal.

Warning signs:
  ‖∇h_r‖ / ‖∇h_1‖ > 100   → exploding, increase α or reduce r
  ‖∇h_r‖ / ‖∇h_1‖ < 2     → vanishing, check re-injection
  ‖∇h_t‖ spikes → NaN risk, check clip
```

---

## 7. Stability Monitoring

### 7.1 Per-Loop Stats to Track

```rust
struct LoopStats {
    loop_idx: usize,
    hidden_var: f32,        // Var(h_t)
    hidden_norm: f32,       // ‖h_t‖₂
    grad_norm: f32,         // ‖∇h_t‖₂
    expert_load: [u32; E],  // count per expert
    edge_load: [u32; M],    // count per edge
    exit_rate: f32,         // inference only
}
```

### 7.2 Stability Thresholds

```
┌──────────────────────┬──────────────┬─────────────────────┐
│  Metric              │  Normal      │  Alert              │
├──────────────────────┼──────────────┼─────────────────────┤
│  Var(h_t)            │  ~ Var(h_0)  │  > 10× Var(h_0)     │
│  ‖h_t‖₂              │  bounded     │  > 5× ‖h_0‖₂        │
│  grad_norm           │  < 10        │  > 100 or NaN       │
│  max expert_load     │  < 3× mean   │  > 5× mean          │
│  max edge_load       │  < 3× mean   │  > 5× mean          │
│  exit_rate (infer)   │  30–50%      │  < 10% or > 90%     │
└──────────────────────┴──────────────┴─────────────────────┘
```

### 7.3 Automatic Interventions

```
If Var(h_t) grows too fast:
  → reduce α (0.5 → 0.3)
  → increase β (0.1 → 0.2)
  → clip hidden states: h = h.clamp(-10, 10)

If expert_load imbalanced:
  → increase λ₁ (0.01 → 0.03)
  → add expert dropout (random expert skip)

If edge_load imbalanced:
  → increase λ₂ (0.01 → 0.03)
  → add edge dropout

If grad_norm > threshold:
  → clip (already doing, but lower to 0.5)
  → reduce lr
```

### 7.4 W&B / TensorBoard Logs

```
Essential charts:
  • loss/loss_total
  • loss/loss_ce
  • loss/loss_aux
  • loss/loss_edge
  • lr
  • grad_norm
  • stability/var_h_t (per loop, overlay)
  • stability/norm_h_t (per loop, overlay)
  • expert/load_histogram
  • edge/load_histogram
  • throughput/tokens_per_sec
  • memory/peak_gb
```

---

## 8. Memory Optimization

### 8.1 VRAM Budget (RTX 4090, 24GB)

```
┌──────────────────────────┬──────────────┐
│  Component               │  VRAM        │
├──────────────────────────┼──────────────┤
│  Model weights (bf16)    │  ~800 MB     │
│  Expert weights (bf16)   │  ~400 MB     │
│  Optimizer states (fp32) │  ~2.4 GB     │
│  Activations (ckpt)      │  ~1 GB       │
│  Gradients               │  ~800 MB     │
│  Data batch              │  ~128 MB     │
├──────────────────────────┼──────────────┤
│  Total                   │  ~5.5 GB     │
│  Available               │  24 GB       │
│  Headroom                │  ✓           │
└──────────────────────────┴──────────────┘

Note: heavy optimizations allow bigger batches.
```

### 8.2 Optimization Checklist

- [ ] Gradient checkpointing (mandatory for r ≥ 6)
- [ ] BF16 mixed precision
- [ ] FlashAttention-2 for attention
- [ ] FSDP if model > 1B (multi-GPU)
- [ ] CPU offload optimizer states (if OOM)
- [ ] 8-bit Adam (bitsandbytes) if tight on memory
- [ ] Micro-batch + grad accumulation

### 8.3 Micro-Batching

```
Target:  batch = 32
If OOM at batch=32:

  micro_batch = 8,  grad_accum = 4
  → effective batch = 32
  → same gradients
  → 4× smaller peak memory
```

---

## 9. Checkpointing

### 9.1 What to Save

```
checkpoint/
├── model.safetensors        # model weights
├── optimizer.pt             # AdamW state
├── scheduler.pt             # lr schedule state
├── step.txt                 # current step
├── config.yaml              # model config
└── train_config.yaml        # training config

Size: ~5 GB total
```

### 9.2 Save Strategy

```
┌─────────────────────┬─────────────────────┐
│  When              │  What to save       │
├─────────────────────┼─────────────────────┤
│  Every 5000 steps   │  Full checkpoint    │
│  Every 500 steps    │  Model only         │
│  On divergence      │  For debugging      │
│  Best eval          │  Full checkpoint    │
└─────────────────────┴─────────────────────┘
```

### 9.3 Resume Flow

```bash
# Resume from checkpoint
spider train \
  --config configs/train-pretrain.yaml \
  --resume checkpoints/step_50000/

# Verify:
#   1. Load model weights
#   2. Load optimizer state
#   3. Load scheduler state
#   4. Set step counter
#   5. Continue dataloader from same position
```

### 9.4 Safetensors Format

```
Why safetensors?
  • Memory-mapped load (fast)
  • No code execution on load (secure)
  • Framework-agnostic
  • Alignment-friendly

Structure:
  model.safetensors
    ├── model.embed.weight         [V, H]
    ├── model.prelude.0.attn.Q     [H, H]
    ├── ...
    ├── model.block.experts.0.W_up [H, H_up]
    └── model.head.weight          [H, V]

Metadata:
  { "format": "pt", "version": "1.0", "step": 50000 }
```

---

## 10. Logging & Metrics

### 10.1 Log Cadence

```
Every step:
  • loss
  • lr
  • grad_norm (running mean)

Every 10 steps:
  • per-loop variance
  • expert load (histogram)
  • edge load (histogram)
  • tokens/sec

Every 1000 steps:
  • validation loss
  • sample generation (2–3 prompts)
  • perplexity

Every 5000 steps:
  • full checkpoint
  • ablation snapshot
```

### 10.2 Validation

```
val_loss on held-out set:
  • Same distribution as train
  • 1000 batches
  • No dropout, no grad

Sample generations:
  • 3 fixed prompts
  • Temperature 0.7, top-p 0.9
  • Save to logs/samples/step_XXXXX.txt
  • Manually inspect — critical for detecting degeneration
```

### 10.3 Console Log Format

```
step 1000 | loss 5.234 | ce 5.180 | aux 1.42 | edge 1.51
            lr 2.85e-4 | gnorm 2.31 | tok/s 8421 | mem 5.8GB
            var_h [1.00, 1.03, 1.07, 1.12, 1.18, 1.24]
            expert_load [0.14, 0.13, 0.12, 0.13, 0.12, 0.12, 0.12, 0.12]
```

**Reading:**
- `var_h` growing slowly → OK (1.0 → 1.24 over 6 loops = stable)
- `expert_load` roughly uniform → good
- If `var_h` jumps 3× → instability alert

---

## 11. Recipes

### 11.1 Recipe A: First Working Model (100M)

```
Goal: Verify pipeline works, get a loss curve.

Data:     100M tokens, single domain (code)
Model:    H=256, E=4, k=2, m=2, r=4
Steps:    5,000
Batch:    16 × 512
LR:       3e-4, warmup 500
Time:     ~4 hours on 4090

Expected:
  L_CE: 10.4 → 4.5
  L_aux: 8.0 → 1.2
  Model generates coherent 10-token continuations

Why this first:
  • Fast iteration (4h not 4 days)
  • Debug all bugs before scaling
  • Validates whole pipeline
```

### 11.2 Recipe B: Personal Model (400M)

```
Goal: Usable personal LLM.

Data:     10B tokens, mixed (60/20/10/10)
Model:    H=512, E=8, k=2, m=3, r=6
Steps:    150,000
Batch:    32 × 1024 (grad_accum 4)
LR:       3e-4, warmup 2000, cosine to 3e-5
Time:     ~72 hours on 4090

Expected:
  L_CE: 10.4 → 2.9
  Perplexity: ~18 (val set)
  Strong code, decent reasoning

SFT after:
  Steps: 3,000, LR 5e-5
  Data:  100M instruction tokens
  L_CE: 1.8 → 1.2
```

### 11.3 Recipe C: Ablation (research)

```
Goal: Prove Spider-Web helps over baselines.

Runs (all H=256, 5K steps):
  1. Dense baseline (no MoE, no loop)
  2. Vanilla MoE (no loop, no graph)
  3. Looped dense (no MoE, no graph)
  4. Looped MoE (no graph)
  5. Spider-Web (all)

Compare:
  • Loss at 5K
  • Effective depth vs params
  • Expert utilization
  • Edge utilization

Cost: 5 × 4h = 20h on 4090

Key ablation:
  Run 4 vs Run 5 → does graph help?
  If no → drop graph, simplify.
```

### 11.4 Recipe D: Scaling Up (1B)

```
After Recipe B works:

Change:
  H: 512 → 1024
  E: 8 → 16
  r: 6 → 6
  Batch: 32 → 16 (OOM)
  LR: 3e-4 → 2e-4 (larger model)
  Steps: 150K → 100K (bigger model converges faster/token)

Data: 30B tokens
Time: ~250 hours on 4090

Or: rent 4× A100 for distributed.
```

---

## 12. Troubleshooting

### 12.1 Loss Not Decreasing

```
Symptoms:
  L_CE stuck > 8.0 after 1K steps

Diagnose:
  1. Check lr — too low? (try 1e-3 for 500 steps)
  2. Check data — random tokens? (should be ~10.4 = ln(32K))
  3. Check gradient — ‖∇θ‖≈0? (dead params)
  4. Check init — W matrices scaled right?

Fix:
  • Lower lr to 1e-4 for stability probe
  • Verify one token: model should predict it > random
  • Run on tiny dataset (1MB) — should overfit in 100 steps
```

### 12.2 Loss Spikes

```
Symptoms:
  L_CE suddenly jumps (e.g., 3.5 → 8.0)

Diagnose:
  1. Check var_h — one loop exploded?
  2. Check grad_norm — before spike?
  3. Check batch — outlier sample?

Fix:
  • Lower lr temporarily (restore after 100 steps)
  • Increase grad_clip (1.0 → 0.5)
  • Skip the offending batch
  • If recurrent: reduce r or α
```

### 12.3 Expert Collapse

```
Symptoms:
  L_aux > 3.0, expert_load concentrated on 1–2 experts

Diagnose:
  Check expert_load histogram — should be uniform.
  If one expert > 50% tokens → collapse.

Fix:
  • Increase λ₁: 0.01 → 0.03
  • Add expert dropout (5%)
  • Add noise to router logits during training
  • Reduce top-k → forces sharing
```

### 12.4 OOM (Out of Memory)

```
Symptoms:
  CUDA OOM at batch or forward

Fix priority:
  1. Reduce micro-batch: 32 → 8
  2. Increase grad_accum: 4 → 16
  3. Enable grad checkpointing (already on?)
  4. Reduce seq_len: 1024 → 512
  5. Use 8-bit Adam (bitsandbytes)
  6. CPU offload optimizer
  7. Reduce r: 6 → 4 (last resort)
```

### 12.5 Degenerate Generation

```
Symptoms:
  Samples repeat tokens, output "the the the the"

Diagnose:
  1. Sample from raw pretrain, no sampling params
  2. Check val_loss — high = model undertrained
  3. Check temperature (should be 0.7, not 1.5)

Fix:
  • Train longer (undertrained is most common)
  • Add repetition penalty (1.1×) at inference
  • Check data quality (washed out text?)
  • Verify tokenizer roundtrip
```

### 12.6 NaN Loss

```
Symptoms:
  L_CE = NaN

Diagnose:
  Usually numerical instability. Trace:
  1. Which step first NaN?
  2. Which loss term? (CE, aux, or edge)
  3. Which layer? (hook forward/backward)

Fix:
  • Lower lr 10×
  • Increase warmup
  • Ensure FP32 softmax in router
  • Check LayerNorm epsilon (1e-5)
  • Add NaN guard: if loss.isnan(): skip_step()
```

---

## 13. Reproducibility

### 13.1 Seeds

```yaml
seed: 42
dataloader_seed: 42
dropout_seed: 42
init_seed: 42
```

### 13.2 Determinism Trade-offs

```
Full determinism:
  • Set PYTHONHASHSEED=0
  • torch.use_deterministic_algorithms(True)
  • cudnn.deterministic = True
  • cudnn.benchmark = False
  Cost: 15-20% slower

Partial determinism (recommended):
  • Fixed seeds for init and data
  • Non-deterministic CUDA ops (faster)
  • Small run-to-run variance (acceptable)
```

### 13.3 Environment

```
Record in checkpoint:
  • Cargo version
  • Rust toolchain
  • CUDA version
  • cuDNN version
  • GPU driver
  • Python/PyTorch version (if used)
  • OS
  • Hardware (GPU name, count)
```

### 13.4 Config Snapshot

```
Save full train_config.yaml per checkpoint.
Enables exact reproduction:
  spider train --resume ckpt/step_50000 --config ckpt/train_config.yaml
```

---

## 14. References

1. **Chinchilla** — Compute-optimal training
2. **LoRA** — Low-rank adaptation (for personal fine-tune)
3. **Q-LoRA** — 4-bit + LoRA for small VRAM
4. **AdamW** — Decoupled weight decay
5. **Wandb** — Experiment tracking
6. **FlashAttention-2** — Efficient attention
7. **Gradient Checkpointing** — Memory trade-off
8. **Switch Transformer** — Aux loss original paper

---

## Appendix A: Training File Map

```
crates/spider-trainer/src/
├── lib.rs
├── trainer.rs              → Section 2
├── optimizer.rs            → Section 4.4
├── checkpoint.rs           → Section 9
├── data_loader.rs          → Section 3
├── loss.rs                 → Section 5
├── stability_monitor.rs    → Section 7
├── metrics.rs              → Section 10
└── scheduler.rs            → Section 4.5
```

---

## Appendix B: CLI Reference

```bash
# Pretrain
spider train \
  --config configs/train-pretrain.yaml \
  --data data/tokenized/train \
  --out checkpoints/pretrain

# Resume
spider train \
  --config configs/train-pretrain.yaml \
  --resume checkpoints/pretrain/step_50000

# SFT
spider train \
  --config configs/train-sft.yaml \
  --data data/tokenized/sft \
  --init checkpoints/pretrain/best

# Monitor (separate terminal)
tensorboard --logdir checkpoints/pretrain/logs
```

---

## Appendix C: Stability Quick Reference

```
┌────────────────────────────────────────────────────────┐
│  Every 100 steps, check:                               │
│                                                        │
│  ✓  loss  → smooth descent                             │
│  ✓  var_h → mild growth (1.0 → 1.3 over r loops)       │
│  ✓  grad  → 0.1 – 10 range, no NaN                     │
│  ✓  expert_load → near uniform                         │
│  ✓  edge_load  → near uniform                          │
│                                                        │
│  Immediate action if:                                  │
│  ✗  loss spike > 2×                                    │
│  ✗  var_h > 5× initial                                 │
│  ✗  grad NaN                                           │
│  ✗  one expert > 50% tokens                            │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

*Document version: 1.0 · Target: personal LLM (100M–1B)*
*Status: Ready to use · Recipes validated at 100M scale*