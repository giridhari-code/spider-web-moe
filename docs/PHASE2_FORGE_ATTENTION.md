# Phase 2 — Forge attention + torch-free Orbit generate

## Goal

Orbit generate without `SpiderWebLM.forward` on the hot path:

1. Native causal attention (prelude + coda) — Forge
2. NativeBackend: embed → prelude → MoE loops → coda → lm_head
3. `OrbitEngine` / `OrbitConfig.native_compute=True`
4. CUDA graph hooks for fixed-loop MoE (later)

## Attention (Phase 2.0)

Causal MHA, no RoPE, no KV cache in C yet.

```c
size_t spider_attention_workspace_bytes(int B, int T, int H, int n_heads);

int spider_attention_causal_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    float* out, float* workspace);
```

Phase 2.1: RoPE + incremental KV.

## Orbit

```python
OrbitConfig(native_compute=True)
NativeBackend.from_spiderw(...).generate(ids, max_new_tokens=n)
```

## Success

- `spider_kernel_phase2` test passes
- Finite attention outputs without torch in C path
