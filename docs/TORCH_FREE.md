# Torch-free inference

## Goal

Run Orbit **without installing PyTorch** when using the native path.

## Install

```bash
pip install -e .                       # numpy + yaml only
# or
pip install -r requirements-native.txt

# Build C kernels (required for MoE native ops)
cmake -S . -B build -DSPIDER_BUILD_TESTS=ON && cmake --build build -j
export SPIDER_KERNEL_LIBDIR=$PWD/build
```

Torch only if training / legacy engine:

```bash
pip install -e ".[torch]"
# or
pip install -e ".[all]"
```

## Runtime

```python
from spider_infer.orbit.engine import OrbitEngine
from spider_infer.orbit.config import OrbitConfig

orbit = OrbitEngine.from_spiderw(
    "model.spiderw",
    OrbitConfig(native_compute=True, max_loops=4),
)
out = orbit.native_generate(prompt_ids, max_new_tokens=32)
```

This path uses:
- `spider_infer/orbit/weight_loader.py` (no torch)
- `spider_infer/orbit/native_backend.py` (numpy + ctypes)
- `libspider_kernels.so` / optional `libspider_cuda.so`

## Still needs torch

| Component | Why |
|-----------|-----|
| `spider_moe.model.SpiderWebLM` | Training / reference |
| `spider_infer.engine.LLMEngine` | Continuous batching torch path |
| `scripts/export_native_weights.py` | Loads `.pt` once to write `.spiderw` |
| Forge / DoBU torch hooks | Optional accelerate |

Export is a one-time step on a machine that has torch; serve machines can stay torch-free.
