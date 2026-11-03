"""Active-set compaction — tokens still in orbit for the current loop.

This is the Spider-native counterpart to "batch size" in dense decode.
In Orbit, effective batch size *shrinks* as tokens exit early; remaining
rows are compacted so expert FFNs and routers only see live work.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ActiveSet:
    """Boolean mask + compacted index list for rows still orbiting."""

    mask: torch.Tensor          # [B] bool
    indices: torch.Tensor       # [N_active] long

    @property
    def n_active(self) -> int:
        return int(self.indices.numel())

    @property
    def batch(self) -> int:
        return int(self.mask.shape[0])

    @classmethod
    def all_active(cls, batch: int, device: torch.device) -> "ActiveSet":
        mask = torch.ones(batch, dtype=torch.bool, device=device)
        indices = torch.arange(batch, device=device)
        return cls(mask=mask, indices=indices)

    def drop(self, exited_local: torch.Tensor) -> "ActiveSet":
        """exited_local: bool [n_active] — True means leave orbit."""
        if exited_local.numel() == 0:
            return self
        still = ~exited_local
        new_indices = self.indices[still]
        new_mask = torch.zeros_like(self.mask)
        if new_indices.numel():
            new_mask[new_indices] = True
        return ActiveSet(mask=new_mask, indices=new_indices)


def compact_active(x: torch.Tensor, active: ActiveSet) -> torch.Tensor:
    """Gather only active rows. x: [B, ...] → [N_active, ...]."""
    if active.n_active == active.batch:
        return x
    return x.index_select(0, active.indices)


def scatter_active(
    full: torch.Tensor,
    active: ActiveSet,
    compacted: torch.Tensor,
) -> torch.Tensor:
    """Write compacted rows back into full batch tensor."""
    if active.n_active == active.batch:
        return compacted
    out = full.clone()
    out.index_copy_(0, active.indices, compacted)
    return out
