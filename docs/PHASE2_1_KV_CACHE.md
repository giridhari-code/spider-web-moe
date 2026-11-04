# Phase 2.1 — Incremental KV cache (native attention)

## Goal

Decode without full-sequence recompute:

```text
Prefill (T>1):  write K,V for [0, T)
Decode  (T=1):  attend to prefix + new token; append K,V
```

## Layout

```c
typedef struct SpiderKVCache {
    float* keys;    /* [B, n_heads, max_T, head_dim] */
    float* values;
    int B, n_heads, max_T, head_dim;
    int length;
} SpiderKVCache;
```

## API

- `spider_kv_bytes` / `spider_kv_init` / `spider_kv_reset`
- `spider_attention_causal_kv_f32(x, B, T, H, n_heads, W_qkv, W_o, cache, position, out, workspace)`
- `spider_attention_kv_workspace_bytes(B, T, H)`

Causal: query at `position+i` sees keys `0..position+i`.
Phase 2.1: no RoPE inside kernel (optional host rotate later).

## Test

`spider_kernel_phase2_1` — prefill T=3 then decode pos=3; capacity overflow returns 3.
