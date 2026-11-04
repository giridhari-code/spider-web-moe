"""FlashMLA analogue: paged decode attention + optional latent compression."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice
from spider_infer.forge.attention.triton_kernels import (
    forge_paged_decode,
    get_last_forge_attn_backend,
)


class ForgeAttention:
    """Unified attention entry used by Orbit shells (prelude/coda).

    Forge 0.2: paged_decode uses Triton on CUDA, torch gather otherwise.
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "torch_sdpa+triton_paged"

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        scale = scale if scale is not None else self.config.attention_softmax_scale
        if q.ndim != 4:
            raise ValueError("q must be [B,H,T,D]")
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def paged_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        page_size: int | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Paged KV decode — Triton (0.2) or torch reference."""
        self.calls += 1
        page_size = page_size or self.config.attention_page_size
        scale = scale if scale is not None else self.config.attention_softmax_scale
        return forge_paged_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            page_size=page_size,
            scale=scale,
        )

    def stats(self) -> dict[str, Any]:
        backend, err = get_last_forge_attn_backend()
        return {
            "pillar": "attention",
            "analogue": "FlashMLA",
            "impl": self._impl,
            "backend": self.backend.value,
            "last_kernel": backend,
            "last_error": err,
            "calls": self.calls,
            "mla_flag": self.config.attention_use_mla,
            "forge_version_target": "0.2",
        }


class FlashMLAStyleAttention(ForgeAttention):
    """Named alias matching DeepSeek FlashMLA role in the Forge map."""
    pass
