"""Forge global configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ForgeBackend(str, Enum):
    """Where kernels execute."""

    TORCH = "torch"          # portable reference (always available)
    TRITON = "triton"        # Triton GPU kernels
    CUDA = "cuda"            # custom CUDA / FlashMLA-class
    ASCEND = "ascend"        # Huawei NPU (torch_npu)
    AUTO = "auto"            # pick best available


@dataclass
class ForgeConfig:
    """Master switchboard for the Forge fabric."""

    backend: ForgeBackend = ForgeBackend.AUTO

    attention_enabled: bool = True
    attention_page_size: int = 16
    attention_use_mla: bool = True
    mla_latent_rank: int = 0
    attention_softmax_scale: float | None = None

    gemm_enabled: bool = True
    gemm_tf32: bool = True
    gemm_fp8: bool = False
    gemm_workspace_mb: int = 64

    ep_enabled: bool = True
    ep_async: bool = False
    ep_num_sms_hint: int = 0

    tile_enabled: bool = True
    tile_default_block_m: int = 64
    tile_default_block_n: int = 64
    tile_default_block_k: int = 32

    select_enabled: bool = True
    select_top_k: int = 2
    select_top_dobu: int = 4

    device: str = "cuda"       # "cuda", "cpu", "npu", "npu:0"
    allow_cpu_fallback: bool = True

    extra: dict = field(default_factory=dict)

    def resolve_backend(self) -> ForgeBackend:
        from spider_infer.forge.backends.detect import detect_capabilities, pick_backend

        return pick_backend(self.backend, device_hint=self.device, caps=detect_capabilities())

    def to_dict(self) -> dict:
        return {
            "name": "Forge",
            "backend": self.resolve_backend().value,
            "requested_backend": self.backend.value,
            "attention": self.attention_enabled,
            "gemm": self.gemm_enabled,
            "ep": self.ep_enabled,
            "tile": self.tile_enabled,
            "select": self.select_enabled,
            "select_top_k": self.select_top_k,
            "select_top_dobu": self.select_top_dobu,
            "device": self.device,
        }
