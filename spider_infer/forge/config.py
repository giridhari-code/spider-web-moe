"""Forge global configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ForgeBackend(str, Enum):
    """Where kernels execute."""

    TORCH = "torch"          # portable reference (always available)
    TRITON = "triton"        # Triton GPU kernels
    CUDA = "cuda"            # custom CUDA / FlashMLA-class
    ASCEND = "ascend"        # Huawei NPU path (future)
    AUTO = "auto"            # pick best available


@dataclass
class ForgeConfig:
    """Master switchboard for the Forge fabric.

    Every pillar is always present in the design; backends may degrade to
    TORCH without removing the API surface.
    """

    backend: ForgeBackend = ForgeBackend.AUTO

    # --- Attention (FlashMLA analogue) ---
    attention_enabled: bool = True
    attention_page_size: int = 16
    attention_use_mla: bool = True          # multi-latent style compress path
    attention_softmax_scale: float | None = None

    # --- GEMM (DeepGEMM analogue) ---
    gemm_enabled: bool = True
    gemm_tf32: bool = True
    gemm_fp8: bool = False
    gemm_workspace_mb: int = 64

    # --- Expert parallel (DeepEP analogue) ---
    ep_enabled: bool = True
    ep_async: bool = False                  # overlap dispatch with compute
    ep_num_sms_hint: int = 0                # 0 = auto

    # --- Tile DSL (TileLang analogue) ---
    tile_enabled: bool = True
    tile_default_block_m: int = 64
    tile_default_block_n: int = 64
    tile_default_block_k: int = 32

    # --- Select (DeepSelect analogue) ---
    select_enabled: bool = True
    select_top_k: int = 2                   # router width (Spider top_k)
    select_top_dobu: int = 4                # Dobu GPU residency (separate!)

    # --- Device / sync ---
    device: str = "cuda"
    allow_cpu_fallback: bool = True

    extra: dict = field(default_factory=dict)

    def resolve_backend(self) -> ForgeBackend:
        if self.backend != ForgeBackend.AUTO:
            return self.backend
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    import triton  # noqa: F401
                    return ForgeBackend.TRITON
                except ImportError:
                    return ForgeBackend.CUDA if torch.cuda.is_available() else ForgeBackend.TORCH
        except ImportError:
            pass
        return ForgeBackend.TORCH

    def to_dict(self) -> dict:
        return {
            "name": "Forge",
            "backend": self.resolve_backend().value,
            "attention": self.attention_enabled,
            "gemm": self.gemm_enabled,
            "ep": self.ep_enabled,
            "tile": self.tile_enabled,
            "select": self.select_enabled,
            "select_top_k": self.select_top_k,
            "select_top_dobu": self.select_top_dobu,
            "device": self.device,
        }
