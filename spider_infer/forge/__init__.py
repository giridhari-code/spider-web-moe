"""Forge — compute kernel fabric under Orbit.

Naming stack
------------
  Sentinel  = model product
  Spider    = neural architecture
  Orbit     = serving runtime
  Dobu      = demand-loaded weight chunks
  **Forge** = GPU/NPU kernel + parallel fabric (this package)

DeepInfer / DeepSeek vendor map → Forge modules
----------------------------------------------
  FlashMLA     → forge.attention   (multi-latent / paged decode attention)
  DeepGEMM     → forge.gemm        (high-throughput GEMM for FFN / experts)
  DeepEP       → forge.ep          (expert-parallel token dispatch / combine)
  TileLang     → forge.tile        (portable tiled kernel DSL + emitters)
  DeepSelect   → forge.select      (routing / top-k / top-dobu selection ops)

Nothing is optional in the *architecture*: every pillar has a stable API.
Backends may start as PyTorch reference and grow CUDA/Triton/Ascend plugins.
"""

from .config import ForgeConfig, ForgeBackend
from .registry import ForgeRuntime, get_forge, init_forge
from .attention import ForgeAttention, FlashMLAStyleAttention
from .gemm import ForgeGEMM, DeepGEMMStyleLinear
from .ep import ForgeEP, DeepEPStyleDispatcher, build_dispatch_plan, dispatch_expert_ffn
from .tile import ForgeTile, TileKernelSpec
from .select import ForgeSelect, DeepSelectStyleRouter
from .runtime import ForgeDevice, synchronize_forge
from .backends import detect_capabilities, pick_backend, ascend_available

__all__ = [
    "ForgeConfig",
    "ForgeBackend",
    "ForgeRuntime",
    "get_forge",
    "init_forge",
    "ForgeAttention",
    "FlashMLAStyleAttention",
    "ForgeGEMM",
    "DeepGEMMStyleLinear",
    "ForgeEP",
    "DeepEPStyleDispatcher",
    "build_dispatch_plan",
    "dispatch_expert_ffn",
    "ForgeTile",
    "TileKernelSpec",
    "ForgeSelect",
    "DeepSelectStyleRouter",
    "ForgeDevice",
    "synchronize_forge",
    "detect_capabilities",
    "pick_backend",
    "ascend_available",
    "FORGE_VERSION",
]

FORGE_VERSION = "0.5.0"
