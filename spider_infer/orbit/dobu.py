"""Orbit Dobu — demand-loaded weight chunks for Spider/Sentinel.

Two *different* controls (do not conflate them)
----------------------------------------------
  top_k     (Spider / MoE router)
            How many experts each *token* mixes in the forward pass.
            Purely a compute / quality knob. Lives on ModelConfig.top_k.

  top_dobu  (Orbit Dobu residency)
            How many expert *weight chunks* may stay hot on the compute
            device (GPU LRU working set). Purely a memory knob.
            Idle chunks remain on CPU storage even if the catalogue is huge.

Example
-------
  top_k=2, top_dobu=4
    → every token routes to 2 experts (compute)
    → up to 4 expert weight tensors may remain resident on GPU (memory)
    → a step that touches experts {0,1,2} loads those three; LRU may keep
      a fourth from a previous step warm.

Stack
-----
  Sentinel  = model product
  Spider    = architecture (owns top_k)
  Orbit     = serving runtime
  Dobu      = weight-chunk residency (owns top_dobu)
"""
from __future__ import annotations

from dataclasses import dataclass

from spider_moe.expert_store import (
    ExpertChunk,
    ExpertChunkStore,
    offload_module_list_to_cpu,
)

DobuChunk = ExpertChunk
DobuStore = ExpertChunkStore


@dataclass
class DobuConfig:
    """Orbit Dobu memory policy — independent of router top_k."""

    enabled: bool = True
    """Maps to ModelConfig.offload_experts."""

    top_dobu: int = 4
    """Max expert weight chunks kept hot on the compute device (GPU LRU).

    This is NOT router width. Router width is ModelConfig.top_k.
    """

    storage_device: str = "cpu"
    pin_memory: bool = True

    # Back-compat alias used in older call sites
    @property
    def max_gpu_chunks(self) -> int:
        return self.top_dobu

    def apply_to_model_config(self, model_cfg) -> None:
        """Enable offload + residency only. Does *not* change top_k."""
        model_cfg.offload_experts = bool(self.enabled)
        model_cfg.offload_max_gpu_experts = int(self.top_dobu)
        # intentionally no assignment to model_cfg.top_k

    def to_dict(self) -> dict:
        return {
            "name": "Orbit Dobu",
            "enabled": self.enabled,
            "top_dobu": self.top_dobu,
            "top_dobu_role": "gpu_chunk_residency_lru",
            "storage_device": self.storage_device,
            "pin_memory": self.pin_memory,
            "note": "top_k is separate (router); top_dobu is Dobu VRAM working set",
        }


def enable_dobu(model, dobu_config: DobuConfig | None = None) -> DobuStore:
    """Attach Dobu store. Leaves model.cfg.top_k / recurrent.top_k untouched."""
    dobu_config = dobu_config or DobuConfig()
    dobu_config.apply_to_model_config(model.cfg)

    rec = model.recurrent
    # Residency budget only — do not overwrite router top_k
    rec.top_dobu = int(dobu_config.top_dobu)

    store = DobuStore(
        rec.experts,
        storage_device=dobu_config.storage_device,
        compute_device=next(model.parameters()).device,
        max_gpu_experts=int(dobu_config.top_dobu),
        pin_memory=dobu_config.pin_memory,
    )
    offload_module_list_to_cpu(rec.experts)
    rec.offload_experts = True
    rec._expert_store = store
    return store


def dobu_stats(model) -> dict | None:
    rec = getattr(model, "recurrent", None)
    store = getattr(rec, "_expert_store", None) if rec is not None else None
    if store is None:
        return None
    st = store.stats()
    st["name"] = "Orbit Dobu"
    st["top_k"] = getattr(rec, "top_k", None)  # router (Spider)
    st["top_dobu"] = getattr(rec, "top_dobu", None) or st.get("max_gpu_experts")
    st["top_k_role"] = "router_experts_per_token"
    st["top_dobu_role"] = "gpu_chunk_residency_lru"
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
