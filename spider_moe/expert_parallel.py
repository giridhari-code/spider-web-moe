"""Minimal Expert Parallelism over torch.distributed.

Strategy (reference, correct, simple):
- Router + non-expert weights are replicated.
- Expert modules are sharded: rank r owns experts where expert_id % world == r.
- Each rank computes only its local experts' contribution to `delta`, then
  `all_reduce` (SUM) so every rank sees the full delta.

This is not as bandwidth-optimal as DeepEP token dispatch, but it is correct
and works with standard NCCL without custom kernels.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def ep_enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank_world() -> tuple[int, int]:
    if not ep_enabled():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def owns_expert(expert_id: int, rank: int, world: int) -> bool:
    return expert_id % world == rank


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce when EP is active; no-op otherwise."""
    if ep_enabled() and tensor.numel() > 0:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
