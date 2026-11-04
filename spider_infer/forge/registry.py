"""Forge runtime registry — multi-backend fabric."""
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
from .backends.detect import detect_capabilities
from .backends.ascend import AscendOps


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
    capabilities: dict[str, Any] = field(default_factory=dict)
    ascend: AscendOps | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict:
        from spider_infer.forge import FORGE_VERSION

        return {
            "technology": "Forge",
            "version": FORGE_VERSION,
            "config": self.config.to_dict(),
            "backend": self.backend.value,
            "device": str(self.device.compute),
            "capabilities": self.capabilities,
            "ascend": self.ascend.stats() if self.ascend else None,
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
    caps = detect_capabilities()
    backend = config.resolve_backend()

    # Align device string with backend when AUTO-ish
    device_str = config.device
    if backend == ForgeBackend.ASCEND and not device_str.startswith("npu"):
        device_str = "npu:0"
    elif backend in (ForgeBackend.CUDA, ForgeBackend.TRITON) and device_str == "cpu":
        if caps.cuda:
            device_str = "cuda:0"

    device = ForgeDevice.from_string(device_str)
    ascend = AscendOps() if backend == ForgeBackend.ASCEND or caps.ascend else AscendOps()

    rt = ForgeRuntime(
        config=config,
        device=device,
        backend=backend,
        capabilities=caps.to_dict(),
        ascend=ascend,
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
