"""Forge runtime registry — one place Orbit asks for kernels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ForgeConfig, ForgeBackend
from .runtime import ForgeDevice
from .attention import ForgeAttention
from .gemm import ForgeGEMM
from .ep import ForgeEP
from .tile import ForgeTile
from .select import ForgeSelect


@dataclass
class ForgeRuntime:
    """Constructed fabric: all five pillars always attached."""

    config: ForgeConfig
    device: ForgeDevice
    attention: ForgeAttention
    gemm: ForgeGEMM
    ep: ForgeEP
    tile: ForgeTile
    select: ForgeSelect
    backend: ForgeBackend = ForgeBackend.TORCH
    meta: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "technology": "Forge",
            "version": "0.1.0",
            "config": self.config.to_dict(),
            "backend": self.backend.value,
            "device": str(self.device.compute),
            "pillars": {
                "attention": self.attention.stats(),
                "gemm": self.gemm.stats(),
                "ep": self.ep.stats(),
                "tile": self.tile.stats(),
                "select": self.select.stats(),
            },
        }


_GLOBAL: ForgeRuntime | None = None


def init_forge(config: ForgeConfig | None = None) -> ForgeRuntime:
    global _GLOBAL
    config = config or ForgeConfig()
    backend = config.resolve_backend()
    device = ForgeDevice.from_string(config.device)
    rt = ForgeRuntime(
        config=config,
        device=device,
        backend=backend,
        attention=ForgeAttention(config, device, backend),
        gemm=ForgeGEMM(config, device, backend),
        ep=ForgeEP(config, device, backend),
        tile=ForgeTile(config, device, backend),
        select=ForgeSelect(config, device, backend),
    )
    _GLOBAL = rt
    return rt


def get_forge() -> ForgeRuntime:
    global _GLOBAL
    if _GLOBAL is None:
        return init_forge()
    return _GLOBAL
