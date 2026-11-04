"""Forge GEMM — DeepGEMM-class matrix multiply for FFN / expert paths."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice
from spider_infer.forge.gemm.triton_kernels import (
    triton_linear,
    triton_expert_ffn,
    get_last_forge_gemm_backend,
)


class ForgeGEMM:
    """High-throughput GEMM facade.

    Forge 0.2: linear / expert_ffn prefer Triton GEMM on CUDA.
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "triton_gemm+torch_fallback"

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        return triton_linear(x, weight, bias)

    def expert_ffn(
        self,
        x: torch.Tensor,
        w_up: torch.Tensor,
        b_up: torch.Tensor | None,
        w_down: torch.Tensor,
        b_down: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls += 1
        return triton_expert_ffn(x, w_up, b_up, w_down, b_down)

    def stats(self) -> dict[str, Any]:
        backend, err = get_last_forge_gemm_backend()
        return {
            "pillar": "gemm",
            "analogue": "DeepGEMM",
            "impl": self._impl,
            "backend": self.backend.value,
            "last_kernel": backend,
            "last_error": err,
            "calls": self.calls,
            "fp8": self.config.gemm_fp8,
            "forge_version_target": "0.2",
        }


class DeepGEMMStyleLinear(ForgeGEMM):
    """Named alias for DeepGEMM role."""
    pass
