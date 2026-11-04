"""Forge EP — DeepEP-class expert-parallel dispatch / combine (0.4)."""
from __future__ import annotations

from typing import Any, Callable

import torch

from spider_infer.forge.config import ForgeBackend, ForgeConfig
from spider_infer.forge.runtime import ForgeDevice
from spider_moe.expert_parallel import ep_enabled, get_rank_world, owns_expert, all_reduce_sum_
from spider_infer.forge.ep.dispatch import (
    DispatchPlan,
    build_dispatch_plan,
    combine_across_ranks,
    dispatch_expert_ffn,
    dispatch_tokens,
    get_last_ep_dispatch_backend,
)


class ForgeEP:
    """Token dispatch + combine across expert-parallel ranks.

    Forge 0.4:
      dispatch_plan / run_dispatched_ffn — DeepEP-style pack→compute→scatter
      combine — cross-rank SUM (all_reduce today; NVSHMEM later)
    """

    def __init__(self, config: ForgeConfig, device: ForgeDevice, backend: ForgeBackend):
        self.config = config
        self.device = device
        self.backend = backend
        self.calls = 0
        self._impl = "dispatch_combine_v04"

    def enabled(self) -> bool:
        return bool(self.config.ep_enabled) and ep_enabled()

    def rank_world(self) -> tuple[int, int]:
        return get_rank_world()

    def owns(self, expert_id: int) -> bool:
        rank, world = self.rank_world()
        return owns_expert(expert_id, rank, world)

    def combine(self, delta: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return combine_across_ranks(delta)

    def dispatch_meta(self, top_indices: torch.Tensor, num_experts: int) -> dict:
        self.calls += 1
        plan = build_dispatch_plan(
            top_indices,
            torch.ones_like(top_indices, dtype=torch.float32),
            num_experts=num_experts,
            local_only=True,
        )
        rank, world = self.rank_world()
        return {
            "rank": rank,
            "world": world,
            "local_experts": plan.local_expert_ids,
            "assignments": len(plan.assignments),
            "top_indices_shape": list(top_indices.shape),
        }

    def dispatch_plan(
        self,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        num_experts: int,
    ) -> DispatchPlan:
        """Build DeepEP-style token packs for this step."""
        self.calls += 1
        return build_dispatch_plan(
            top_indices, top_weights, num_experts=num_experts, local_only=True
        )

    def run_dispatched_ffn(
        self,
        messages: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        *,
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        num_experts: int,
        expert_fn: Callable | None = None,
    ) -> torch.Tensor:
        """dispatch → expert FFN → combine (+ cross-rank)."""
        self.calls += 1
        return dispatch_expert_ffn(
            messages,
            top_indices,
            top_weights,
            w_up=w_up,
            b_up=b_up,
            w_down=w_down,
            b_down=b_down,
            num_experts=num_experts,
            expert_fn=expert_fn,
        )

    def stats(self) -> dict[str, Any]:
        rank, world = self.rank_world()
        return {
            "pillar": "ep",
            "analogue": "DeepEP",
            "impl": self._impl,
            "backend": self.backend.value,
            "last_kernel": get_last_ep_dispatch_backend(),
            "calls": self.calls,
            "dist_initialized": ep_enabled(),
            "rank": rank,
            "world": world,
            "async": self.config.ep_async,
            "forge_version_target": "0.4",
        }


class DeepEPStyleDispatcher(ForgeEP):
    """Named alias for DeepEP role."""
    pass
