"""Forge 0.4 — DeepEP-style token dispatch / combine.

Pattern (single- and multi-rank):
  1. Router yields top_indices / top_weights per token.
  2. **dispatch**: group tokens that selected expert e (pack contiguous).
  3. Run expert FFN only on packed tokens (and only local experts if EP).
  4. **combine**: scatter weighted outputs back to the full token grid;
     multi-rank: SUM all_reduce so every rank sees full delta.

This mirrors DeepEP's dispatch→compute→combine pipeline without requiring
NVSHMEM; a faster collective can replace ``combine_across_ranks`` later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from spider_moe.expert_parallel import (
    all_reduce_sum_,
    ep_enabled,
    get_rank_world,
    owns_expert,
)

LAST_EP_DISPATCH_BACKEND = "not_run"


@dataclass
class ExpertAssignment:
    expert_id: int
    token_index: torch.Tensor  # [n] int64 flat positions
    weights: torch.Tensor      # [n] routing weights for this expert


@dataclass
class DispatchPlan:
    """Per-step dispatch layout."""

    batch: int
    seq: int
    hidden: int
    top_k: int
    num_experts: int
    assignments: list[ExpertAssignment] = field(default_factory=list)
    local_expert_ids: list[int] = field(default_factory=list)

    def tokens_for(self, expert_id: int) -> ExpertAssignment | None:
        for a in self.assignments:
            if a.expert_id == expert_id:
                return a
        return None


def build_dispatch_plan(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    local_only: bool = True,
) -> DispatchPlan:
    """Build packed assignments from router outputs.

    top_indices / top_weights: [B, T, K] or [N, K]
    """
    global LAST_EP_DISPATCH_BACKEND
    if top_indices.ndim == 3:
        b, t, k = top_indices.shape
        flat_idx = top_indices.reshape(-1, k)
        flat_w = top_weights.reshape(-1, k)
    elif top_indices.ndim == 2:
        flat_idx = top_indices
        flat_w = top_weights
        n, k = flat_idx.shape
        b, t = n, 1
    else:
        raise ValueError("top_indices must be [B,T,K] or [N,K]")

    n_tokens = flat_idx.shape[0]
    rank, world = get_rank_world()
    if local_only and ep_enabled():
        local_ids = [e for e in range(num_experts) if owns_expert(e, rank, world)]
    else:
        local_ids = list(range(num_experts))

    assignments: list[ExpertAssignment] = []
    for eid in local_ids:
        # accumulate weight if expert appears in any of K slots
        token_w = flat_idx.new_zeros(n_tokens, dtype=flat_w.dtype)
        for slot in range(k):
            mask = flat_idx[:, slot] == eid
            token_w = token_w + flat_w[:, slot] * mask.to(flat_w.dtype)
        active = token_w > 0
        if not active.any():
            continue
        positions = active.nonzero(as_tuple=False).flatten()
        assignments.append(
            ExpertAssignment(
                expert_id=int(eid),
                token_index=positions,
                weights=token_w[positions],
            )
        )

    LAST_EP_DISPATCH_BACKEND = "dispatch_plan_v04"
    return DispatchPlan(
        batch=b,
        seq=t,
        hidden=0,
        top_k=k,
        num_experts=num_experts,
        assignments=assignments,
        local_expert_ids=local_ids,
    )


def dispatch_tokens(
    messages: torch.Tensor,
    assignment: ExpertAssignment,
) -> torch.Tensor:
    """Gather packed tokens for one expert. messages: [B,T,H] or [N,H]."""
    flat = messages.reshape(-1, messages.shape[-1])
    return flat.index_select(0, assignment.token_index)


def combine_tokens(
    delta_flat: torch.Tensor,
    assignment: ExpertAssignment,
    expert_out: torch.Tensor,
) -> None:
    """Scatter-add weighted expert outputs into flat delta (in-place)."""
    weighted = expert_out * assignment.weights.unsqueeze(-1).to(expert_out.dtype)
    delta_flat.index_add_(0, assignment.token_index, weighted)


def combine_across_ranks(delta: torch.Tensor) -> torch.Tensor:
    """DeepEP combine stage across EP ranks (SUM)."""
    if ep_enabled():
        all_reduce_sum_(delta)
    return delta


@torch.no_grad()
def dispatch_expert_ffn(
    messages: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    num_experts: int,
    expert_fn=None,
) -> torch.Tensor:
    """Full dispatch → local expert FFN → combine pipeline.

    w_up: [E, H_up, H], b_up: [E, H_up], w_down: [E, H, H_up], b_down: [E, H]
    messages: [B, T, H]
    Returns delta [B, T, H]
    """
    global LAST_EP_DISPATCH_BACKEND
    b, t, h = messages.shape
    plan = build_dispatch_plan(
        top_indices, top_weights, num_experts=num_experts, local_only=True
    )
    plan.hidden = h
    delta_flat = messages.reshape(-1, h).new_zeros(b * t, h)

    for asg in plan.assignments:
        eid = asg.expert_id
        toks = dispatch_tokens(messages, asg)
        if expert_fn is not None:
            out = expert_fn(eid, toks)
        else:
            up = F.linear(toks, w_up[eid], b_up[eid])
            up = F.gelu(up)
            out = F.linear(up, w_down[eid], b_down[eid])
        combine_tokens(delta_flat, asg, out)

    delta = delta_flat.view(b, t, h)
    combine_across_ranks(delta)
    LAST_EP_DISPATCH_BACKEND = "dispatch_ffn_v04"
    return delta


def get_last_ep_dispatch_backend() -> str:
    return LAST_EP_DISPATCH_BACKEND
