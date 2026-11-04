"""Forge GEMM — DeepGEMM-class matmul; 0.3 adds FP8 weight path."""
from __future__ import annotations

from typing import Any

import torch

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice
from spider_infer.forge.gemm.triton_kernels import (
    triton_linear,
    triton_expert_ffn,
    get_last_forge_gemm_backend,
)
from spider_infer.forge.gemm.fp8 import (
    fp8_linear,
    fp8_expert_ffn,
    get_last_fp8_gemm_backend,
)


class ForgeGEMM:
    """0.2 Triton linear · 0.3 FP8 weight linear when config.gemm_fp8."""

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "triton+fp8+torch"

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        if self.config.gemm_fp8:
            return fp8_linear(x, weight, bias)
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
        if self.config.gemm_fp8:
            return fp8_expert_ffn(x, w_up, b_up, w_down, b_down)
        return triton_expert_ffn(x, w_up, b_up, w_down, b_down)

    def linear_fp8(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        quantize_input: bool = False,
    ) -> torch.Tensor:
        """Explicit FP8 path regardless of config.gemm_fp8."""
        self.calls += 1
        return fp8_linear(x, weight, bias, quantize_input=quantize_input)

    def stats(self) -> dict[str, Any]:
        t_b, t_e = get_last_forge_gemm_backend()
        f_b, f_e = get_last_fp8_gemm_backend()
        return {
            "pillar": "gemm",
            "analogue": "DeepGEMM",
            "impl": self._impl,
            "backend": self.backend.value,
            "last_triton_kernel": t_b,
            "last_fp8_kernel": f_b,
            "last_error": t_e or f_e,
            "calls": self.calls,
            "fp8": self.config.gemm_fp8,
            "forge_version_target": "0.3",
        }


class DeepGEMMStyleLinear(ForgeGEMM):
    pass
