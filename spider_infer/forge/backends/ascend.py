"""Ascend NPU backend hooks (Forge 0.5).

When ``torch_npu`` is installed, ops prefer NPU device placement.
Kernel bodies still use portable torch math until CANN-specific ops are linked;
the API and device routing are production-shaped so Orbit does not branch.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def ascend_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False


def ascend_device(index: int = 0) -> torch.device:
    if not ascend_available():
        return torch.device("cpu")
    return torch.device(f"npu:{index}")


class AscendOps:
    """Thin op surface for Ascend — mirrors ForgeGEMM / attention needs."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.available = ascend_available()
        self.calls = 0

    @property
    def device(self) -> torch.device:
        return ascend_device(self.device_index)

    def to_device(self, t: torch.Tensor) -> torch.Tensor:
        if self.available and t.device.type != "npu":
            return t.to(self.device)
        return t

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        if self.available:
            x = self.to_device(x)
            weight = self.to_device(weight)
            if bias is not None:
                bias = self.to_device(bias)
        return F.linear(x, weight, bias)

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        if self.available:
            q, k, v = self.to_device(q), self.to_device(k), self.to_device(v)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "ascend",
            "available": self.available,
            "device": str(self.device),
            "calls": self.calls,
            "impl": "torch_npu_portable" if self.available else "unavailable_fallback_cpu",
        }
