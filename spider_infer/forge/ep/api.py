"""Forge EP — DeepEP-class expert-parallel dispatch / combine."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice
from spider_moe.expert_parallel import ep_enabled, get_rank_world, owns_expert, all_reduce_sum_


class ForgeEP:
    """Token dispatch + combine across expert-parallel ranks.

    DeepEP provides optimized all-to-all; this API is the stable Orbit surface.
    Current body: ownership mask + SUM all_reduce (correct, simple).
    Future body: NVSHMEM / DeepEP-style dispatch without API change.
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "allreduce_sum"

    def enabled(self) -> bool:
        return bool(self.config.ep_enabled) and ep_enabled()

    def rank_world(self) -> tuple[int, int]:
        return get_rank_world()

    def owns(self, expert_id: int) -> bool:
        rank, world = self.rank_world()
        return owns_expert(expert_id, rank, world)

    def combine(self, delta: torch.Tensor) -> torch.Tensor:
        """Sum partial expert outputs across EP ranks (in-place when possible)."""
        self.calls += 1
        if self.enabled():
            all_reduce_sum_(delta)
        return delta

    def dispatch_meta(self, top_indices: torch.Tensor, num_experts: int) -> dict:
        """Describe which local experts see which tokens (planning aid)."""
        self.calls += 1
        rank, world = self.rank_world()
        local = [e for e in range(num_experts) if owns_expert(e, rank, world)]
        return {
            "rank": rank,
            "world": world,
            "local_experts": local,
            "top_indices_shape": list(top_indices.shape),
        }

    def stats(self) -> dict[str, Any]:
        rank, world = self.rank_world()
        return {
            "pillar": "ep",
            "analogue": "DeepEP",
            "impl": self._impl,
            "backend": self.backend.value,
            "calls": self.calls,
            "dist_initialized": ep_enabled(),
            "rank": rank,
            "world": world,
            "async": self.config.ep_async,
        }


class DeepEPStyleDispatcher(ForgeEP):
    """Named alias for DeepEP role."""
    pass
