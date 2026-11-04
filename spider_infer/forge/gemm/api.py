"""Forge GEMM — DeepGEMM-class matrix multiply for FFN / expert paths."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice


class ForgeGEMM:
    """High-throughput GEMM facade.

    DeepGEMM maps here: same call sites for expert up/down projections;
    CUDA/FP8 plugins swap the body without changing Orbit/Dobu callers.
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "torch_addmm"

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x @ weight.T + bias  (weight: [out, in])."""
        self.calls += 1
        return F.linear(x, weight, bias)

    def expert_ffn(
        self,
        x: torch.Tensor,
        w_up: torch.Tensor,
        b_up: torch.Tensor | None,
        w_down: torch.Tensor,
        b_down: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard expert FFN: GELU(x W_up) W_down."""
        self.calls += 1
        h = F.linear(x, w_up, b_up)
        h = F.gelu(h)
        return F.linear(h, w_down, b_down)

    def stats(self) -> dict[str, Any]:
        return {
            "pillar": "gemm",
            "analogue": "DeepGEMM",
            "impl": self._impl,
            "backend": self.backend.value,
            "calls": self.calls,
            "fp8": self.config.gemm_fp8,
        }


class DeepGEMMStyleLinear(ForgeGEMM):
    """Named alias for DeepGEMM role."""
    pass
