"""Orbit 0.3 — CUDA graph policy keyed by (A, length_bucket, L).

A = batch / active size at decode step start (early_exit must be off).
L = fixed recurrent loop count (model.cfg.num_loops).
"""
from __future__ import annotations

from spider_infer.cuda_graph import CudaGraphDecodeRunner, DecodeGraphKey

__all__ = ["CudaGraphDecodeRunner", "DecodeGraphKey", "orbit_graph_allowed"]


def orbit_graph_allowed(*, early_exit: bool, device_type: str) -> bool:
    """Static graphs require fixed control flow and CUDA."""
    return (not early_exit) and device_type == "cuda"
