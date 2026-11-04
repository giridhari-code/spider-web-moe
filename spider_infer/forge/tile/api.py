"""Forge Tile — TileLang-class portable tiled kernel descriptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice


@dataclass
class TileKernelSpec:
    """Declarative tile problem (TileLang analogue IR-lite)."""

    name: str
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    dtype: str = "float16"
    meta: dict[str, Any] = field(default_factory=dict)


class ForgeTile:
    """Registers and runs tiled ops; emitters target Triton/CUDA later.

    TileLang in DeepSeek's stack is a kernel language. Forge Tile keeps an
    explicit spec registry so Orbit never hard-codes block sizes in MoE code.
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._specs: dict[str, TileKernelSpec] = {}
        self._impls: dict[str, Callable] = {}
        self._impl = "spec_registry"
        # Default fused expert tile
        self.register(
            TileKernelSpec(
                name="expert_ffn",
                block_m=config.tile_default_block_m,
                block_n=config.tile_default_block_n,
                block_k=config.tile_default_block_k,
            )
        )

    def register(self, spec: TileKernelSpec, impl: Callable | None = None) -> None:
        self._specs[spec.name] = spec
        if impl is not None:
            self._impls[spec.name] = impl

    def get_spec(self, name: str) -> TileKernelSpec:
        if name not in self._specs:
            raise KeyError(f"unknown tile spec {name!r}")
        return self._specs[name]

    def run(self, name: str, *args, **kwargs):
        self.calls += 1
        if name in self._impls:
            return self._impls[name](*args, **kwargs)
        raise RuntimeError(
            f"tile {name!r} has no backend impl yet — register an impl or use ForgeGEMM"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "pillar": "tile",
            "analogue": "TileLang",
            "impl": self._impl,
            "backend": self.backend.value,
            "calls": self.calls,
            "specs": list(self._specs.keys()),
        }
