"""FlashMLA analogue: paged decode attention + optional latent compression."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice


class ForgeAttention:
    """Unified attention entry used by Orbit shells (prelude/coda).

    Backends
    --------
    TORCH  — SDPA reference
    TRITON/CUDA — plug FlashMLA-style kernels when compiled (same API)
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "torch_sdpa"

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Single-step or short-query attention. q/k/v: [B,H,T,D] or [B,T,H,D] via SDPA."""
        self.calls += 1
        scale = scale if scale is not None else self.config.attention_softmax_scale
        # Expect [B, H, T, D] for SDPA
        if q.ndim != 4:
            raise ValueError("q must be [B,H,T,D]")
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=scale
        )
        return out

    def paged_decode(
        self,
        q: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        page_size: int | None = None,
    ) -> torch.Tensor:
        """Paged KV decode — architecture stable; body may call Triton path later.

        For now gathers pages then SDPA (correctness-first, FlashMLA-shaped API).
        """
        self.calls += 1
        page_size = page_size or self.config.attention_page_size
        # q: [B, H, 1, D]; cache: [num_pages, H, page_size, D] simplified layout
        # Reference gather — production FlashMLA replaces this body.
        b, h, tq, d = q.shape
        if tq != 1:
            # fall back to dense path if caller packed full K/V
            return self.decode(q, cache_k, cache_v, causal=False)
        outs = []
        for i in range(b):
            length = int(seq_lens[i].item())
            n_blocks = (length + page_size - 1) // page_size
            pieces_k = []
            pieces_v = []
            for bi in range(n_blocks):
                page = int(block_tables[i, bi].item())
                if page < 0:
                    break
                pieces_k.append(cache_k[page])
                pieces_v.append(cache_v[page])
            if not pieces_k:
                outs.append(q[i : i + 1].zeros_like(q[i : i + 1]))
                continue
            k = torch.cat(pieces_k, dim=1)[:, :length]  # [H, L, D]
            v = torch.cat(pieces_v, dim=1)[:, :length]
            qi = q[i]  # [H, 1, D]
            oi = F.scaled_dot_product_attention(
                qi.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False
            )
            outs.append(oi)
        return torch.cat(outs, dim=0)

    def stats(self) -> dict[str, Any]:
        return {
            "pillar": "attention",
            "analogue": "FlashMLA",
            "impl": self._impl,
            "backend": self.backend.value,
            "calls": self.calls,
            "mla_flag": self.config.attention_use_mla,
        }


class FlashMLAStyleAttention(ForgeAttention):
    """Named alias matching DeepSeek FlashMLA role in the Forge map."""
    pass
