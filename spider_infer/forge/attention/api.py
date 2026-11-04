"""FlashMLA analogue: paged decode + MLA latent path (Forge 0.3)."""
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
from spider_infer.forge.attention.mla import mla_decode, get_last_mla_backend


class ForgeAttention:
    """Orbit shells attention entry.

    0.2 paged_decode (Triton/torch)
    0.3 decode_mla — FlashMLA-style latent path
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "paged+mla"

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
        if self.config.attention_use_mla:
            rank = int(getattr(self.config, "mla_latent_rank", 0) or 0) or None
            return mla_decode(
                q, k, v, latent_rank=rank, scale=scale, causal=causal
            )
        scale = scale if scale is not None else self.config.attention_softmax_scale
        if q.ndim != 4:
            raise ValueError("q must be [B,H,T,D]")
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def decode_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        latent_rank: int | None = None,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Explicit FlashMLA-style entry (always latent path)."""
        self.calls += 1
        return mla_decode(
            q, k, v, latent_rank=latent_rank, scale=scale, causal=causal
        )

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
        self.calls += 1
        page_size = page_size or self.config.attention_page_size
        scale = scale if scale is not None else self.config.attention_softmax_scale
        return forge_paged_decode(
            q, k_cache, v_cache, block_table, seq_lens,
            page_size=page_size, scale=scale,
        )

    def stats(self) -> dict[str, Any]:
        paged_b, paged_e = get_last_forge_attn_backend()
        mla_b, mla_e = get_last_mla_backend()
        return {
            "pillar": "attention",
            "analogue": "FlashMLA",
            "impl": self._impl,
            "backend": self.backend.value,
            "last_paged_kernel": paged_b,
            "last_mla_kernel": mla_b,
            "last_error": paged_e or mla_e,
            "calls": self.calls,
            "mla_flag": self.config.attention_use_mla,
            "forge_version_target": "0.3",
        }


class FlashMLAStyleAttention(ForgeAttention):
    pass
