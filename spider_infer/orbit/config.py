"""Orbit runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrbitConfig:
    """Knobs that make Orbit Spider-aware rather than dense-Transformer-aware."""

    max_loops: int = 4
    early_exit: bool = True
    exit_threshold: float = 0.85
    prefer_decode: bool = True
    track_orbit_stats: bool = True
    page_size: int = 16
    max_orbit_work: int = 0
    enable_cuda_graph: bool = True
    graph_length_bucket: int = 64
    draft_tokens: int = 0
    architecture: str = "spider"
    dobu_enabled: bool = False
    dobu_max_gpu_chunks: int = 4
    top_dobu: int = 4
    forge_enabled: bool = True
    # Phase 2: run MoE (+ optional native attn) without torch.nn forward
    native_compute: bool = False
    spiderw_path: str | None = None

    def validate(self) -> "OrbitConfig":
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not (0.0 < self.exit_threshold < 1.0):
            raise ValueError("exit_threshold must be in (0, 1)")
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.max_orbit_work < 0:
            raise ValueError("max_orbit_work must be >= 0")
        if self.native_compute and not self.spiderw_path:
            # path may be attached later via from_spiderw
            pass
        return self
