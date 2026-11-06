# Phase 2.3 — OrbitEngine.from_spiderw + named weight sections

## Goal

End-to-end native serve entry:

```python
from spider_infer.orbit.engine import OrbitEngine
from spider_infer.orbit.config import OrbitConfig

orbit = OrbitEngine.from_spiderw(
    "model.spiderw",
    OrbitConfig(native_compute=True, max_loops=4),
)
ids = orbit.native_generate(prompt_ids, max_new_tokens=32)
```

## Named sections in `.spiderw`

Section id **200** = named tensor:

```text
payload = uint16 name_len | name utf-8 | raw array bytes
```

Used for `prelude.{i}.W_qkv`, `coda.{i}.W_o`, `edge_value`, etc.

## Export

```bash
python scripts/export_native_weights.py --checkpoint ckpt.pt --out model.spiderw --build-csr-placeholder
```

Exports MoE + optional prelude/coda Linear weights when present in the state dict.
