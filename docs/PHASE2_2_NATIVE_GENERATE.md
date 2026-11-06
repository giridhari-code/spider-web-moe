# Phase 2.2 — NativeBackend generate + multi-layer KV

## Goal

Torch-free greedy decode:

```text
ids → embed → [prelude attn+ffn]* → MoE loops → [coda attn+ffn]* → lm_head → argmax
```

Each attention layer owns a `SpiderKVCache`-compatible NumPy buffer (or C cache via ctypes).

## API

```python
backend = NativeBackend.from_spiderw("model.spiderw")
out_ids = backend.generate(prompt_ids, max_new_tokens=32)
```

- Prefill: full prompt through layers, fill KV
- Decode: one token at a time, append KV
- If prelude/coda QKV weights absent in `.spiderw`, shells are **identity** (MoE-only path)

## Weight keys (optional sections)

| Key | Shape |
|-----|--------|
| `prelude.{i}.W_qkv` | `[3H, H]` |
| `prelude.{i}.W_o` | `[H, H]` |
| `prelude.{i}.W_ffn_up` / `W_ffn_down` | optional |
| `coda.{i}.*` | same |

Export script can add these in a follow-up.

## Files

- `spider_infer/orbit/native_backend.py` — `LayerKV`, `generate`, `forward_token`
- `docs/PHASE2_2_NATIVE_GENERATE.md`
