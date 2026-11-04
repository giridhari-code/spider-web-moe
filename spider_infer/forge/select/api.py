"""Forge Select — DeepSelect-class routing / top-k / top-dobu selection."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice


class ForgeSelect:
    """Router selection ops with explicit top_k vs top_dobu.

    DeepSelect optimises selection kernels; this API is the Orbit/Dobu contract:
      top_k     → how many experts mix per token (compute)
      top_dobu  → GPU chunk residency budget (memory) — reported, not used as k
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "torch_topk"

    def route(
        self,
        logits: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """logits [..., E] → (top_weights, top_indices) with softmax over top-k."""
        self.calls += 1
        k = int(top_k if top_k is not None else self.config.select_top_k)
        k = max(1, min(k, logits.shape[-1]))
        vals, idx = torch.topk(logits, k=k, dim=-1)
        weights = F.softmax(vals, dim=-1)
        return weights, idx

    def top_dobu_budget(self) -> int:
        """Residency budget for Dobu (independent of top_k)."""
        return int(self.config.select_top_dobu)

    def stats(self) -> dict[str, Any]:
        return {
            "pillar": "select",
            "analogue": "DeepSelect",
            "impl": self._impl,
            "backend": self.backend.value,
            "calls": self.calls,
            "top_k": self.config.select_top_k,
            "top_dobu": self.config.select_top_dobu,
        }


class DeepSelectStyleRouter(ForgeSelect):
    """Named alias for DeepSelect role."""
    pass
