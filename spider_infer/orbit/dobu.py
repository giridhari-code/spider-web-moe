"""Orbit Dobu — demand-loaded weight chunks for Spider/Sentinel.

Name
----
  Orbit  = serving technology
  Dobu   = this optimisation: only the weight slices needed for the current
           input occupy compute memory; the rest stay on host (CPU) storage.

Principle
---------
  A full MoE checkpoint is a catalogue of expert chunks.
  The router decides which chunks a token needs.
  Dobu materialises *only those chunks* on the GPU (small LRU working set),
  runs the FFN, and leaves idle experts off-device.

  Shells (embed, prelude/coda attention, router) stay resident — they are
  always required. Expert FFNs are the bulk of MoE parameters and the win.

Stack parallel
--------------
  Sentinel  = model product
  Spider    = architecture
  Orbit     = serving runtime
  Dobu      = sparse weight residency policy (this module)
"""
from __future__ import annotations

from dataclasses import dataclass

from spider_moe.expert_store import (
    ExpertChunk,
    ExpertChunkStore,
    offload_module_list_to_cpu,
)

# Public aliases — Orbit Dobu vocabulary
DobuChunk = ExpertChunk
DobuStore = ExpertChunkStore


@dataclass
class DobuConfig:
    """Knobs for Orbit Dobu weight residency."""

    enabled: bool = True
    """Master switch — maps to ModelConfig.offload_experts."""

    max_gpu_chunks: int = 4
    """How many expert chunks may stay hot on the compute device (LRU)."""

    storage_device: str = "cpu"
    """Where idle chunks live (usually CPU; future: disk-backed)."""

    pin_memory: bool = True
    """Pin host storage for faster H2D when CUDA is present."""

    def apply_to_model_config(self, model_cfg) -> None:
        model_cfg.offload_experts = bool(self.enabled)
        model_cfg.offload_max_gpu_experts = int(self.max_gpu_chunks)

    def to_dict(self) -> dict:
        return {
            "name": "Orbit Dobu",
            "enabled": self.enabled,
            "max_gpu_chunks": self.max_gpu_chunks,
            "storage_device": self.storage_device,
            "pin_memory": self.pin_memory,
        }


def enable_dobu(model, dobu_config: DobuConfig | None = None) -> DobuStore:
    """Attach / rebuild Dobu store on an existing SpiderWebLM (after .to(device))."""
    dobu_config = dobu_config or DobuConfig()
    rec = model.recurrent
    store = DobuStore(
        rec.experts,
        storage_device=dobu_config.storage_device,
        compute_device=next(model.parameters()).device,
        max_gpu_experts=dobu_config.max_gpu_chunks,
        pin_memory=dobu_config.pin_memory,
    )
    offload_module_list_to_cpu(rec.experts)
    rec.offload_experts = True
    rec._expert_store = store
    return store


def dobu_stats(model) -> dict | None:
    """Return Dobu residency stats, or None if Dobu is not active."""
    rec = getattr(model, "recurrent", None)
    store = getattr(rec, "_expert_store", None) if rec is not None else None
    if store is None:
        return None
    st = store.stats()
    st["name"] = "Orbit Dobu"
    return st


__all__ = [
    "DobuConfig",
    "DobuChunk",
    "DobuStore",
    "ExpertChunk",
    "ExpertChunkStore",
    "enable_dobu",
    "dobu_stats",
    "offload_module_list_to_cpu",
]
