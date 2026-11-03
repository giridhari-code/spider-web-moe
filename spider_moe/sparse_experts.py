"""Sparse expert FFN helpers for SpiderMoELayer."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .expert_parallel import ep_enabled, get_rank_world, owns_expert, all_reduce_sum_


def sparse_expert_delta(
    messages: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    num_experts: int,
    top_k: int,
    expert_parallel: bool = True,
) -> torch.Tensor:
    """Evaluate only top-k selected experts on the tokens that chose them.

    messages:     [B, T, H]
    top_indices:  [B, T, K]
    top_weights:  [B, T, K]
    w_up:         [E, H_up, H]
    b_up:         [E, H_up]
    w_down:       [E, H, H_up]
    b_down:       [E, H]

    Returns delta [B, T, H] numerically equivalent to dense expert mix.
    """
    b, t, width = messages.shape
    e = num_experts

    use_ep = expert_parallel and ep_enabled()
    if use_ep:
        rank, world = get_rank_world()
        local_experts = [eid for eid in range(e) if owns_expert(eid, rank, world)]
    else:
        local_experts = list(range(e))

    flat_msg = messages.reshape(-1, width)
    flat_idx = top_indices.reshape(-1, top_k)
    flat_w = top_weights.reshape(-1, top_k)
    n_tokens = flat_msg.shape[0]
    delta_flat = flat_msg.new_zeros(n_tokens, width)

    for eid in local_experts:
        token_weight = flat_msg.new_zeros(n_tokens)
        for k in range(top_k):
            mask_k = flat_idx[:, k] == eid
            token_weight = token_weight + flat_w[:, k] * mask_k.to(flat_w.dtype)
        active = token_weight > 0
        if not active.any():
            continue
        toks = flat_msg[active]
        up = F.linear(toks, w_up[eid], b_up[eid])
        up = F.gelu(up)
        out = F.linear(up, w_down[eid], b_down[eid])
        delta_flat[active] = delta_flat[active] + out * token_weight[active].unsqueeze(-1)

    delta = delta_flat.view(b, t, width)
    if use_ep:
        all_reduce_sum_(delta)
    return delta
