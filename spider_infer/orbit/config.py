"""Orbit runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrbitConfig:
    """Knobs that make Orbit Spider-aware rather than dense-Transformer-aware.

    Parameters
    ----------
    max_loops:
        Hard ceiling on recurrent orbits per token (usually model.num_loops).
    early_exit:
        Allow tokens to leave orbit when exit_head confidence exceeds threshold.
    exit_threshold:
        Sigmoid threshold on exit_head; higher = stay in orbit longer.
    prefer_decode:
        Scheduler prefers decode orbits over new prefills (latency-friendly).
    track_orbit_stats:
        Record per-step active-set sizes and loops-used histograms.
    page_size:
        KV page size for attention shells (vLLM-style paging, Spider-scoped).
    """

    max_loops: int = 4
    early_exit: bool = True
    exit_threshold: float = 0.85
    prefer_decode: bool = True
    track_orbit_stats: bool = True
    page_size: int = 16
    # Orbit 0.2: scheduler decode budget in (seq × loops) units; 0 = unlimited.
    max_orbit_work: int = 0
    # Orbit 0.3: allow CUDA graph capture when early_exit is False.
    enable_cuda_graph: bool = True
    graph_length_bucket: int = 64
    # Orbit 0.4: speculative draft orbits (0 disables).
    draft_tokens: int = 0
    # spider (default) | dense (minimal dense Transformer, no vLLM)
    architecture: str = "spider"
    # Orbit Dobu: demand-load expert weight chunks (CPU storage → GPU on route)
    dobu_enabled: bool = False
    dobu_max_gpu_chunks: int = 4
    # top-dobu: GPU expert-chunk LRU size (memory). Router width is ModelConfig.top_k.
    top_dobu: int = 4
    # Forge compute fabric (attention/gemm/ep/tile/select)
    forge_enabled: bool = True

    def validate(self) -> "OrbitConfig":
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not (0.0 < self.exit_threshold < 1.0):
            raise ValueError("exit_threshold must be in (0, 1)")
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.max_orbit_work < 0:
            raise ValueError("max_orbit_work must be >= 0")
        return self
