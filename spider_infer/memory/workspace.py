"""Reusable tensor workspace pool — cuts alloc/free churn on hot decode paths.

Acquire a buffer of at least the requested shape; release returns it to the pool.
Oversized buffers are kept (capped) so the next call can reshape a view.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


def _key(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> tuple:
    return (shape, dtype, str(device))


@dataclass
class WorkspacePool:
    """Per-(shape, dtype, device) free lists."""

    max_buffers_per_key: int = 8
    _free: dict = field(default_factory=lambda: defaultdict(list))
    hits: int = 0
    misses: int = 0
    releases: int = 0

    def acquire(
        self,
        shape: tuple[int, ...] | list[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        zero: bool = False,
    ) -> torch.Tensor:
        shape = tuple(shape)
        device = torch.device(device)
        k = _key(shape, dtype, device)
        bucket = self._free[k]
        if bucket:
            self.hits += 1
            t = bucket.pop()
            if zero:
                t.zero_()
            return t
        self.misses += 1
        t = torch.empty(shape, dtype=dtype, device=device)
        if zero:
            t.zero_()
        return t

    def release(self, t: torch.Tensor) -> None:
        if t is None or not torch.is_tensor(t):
            return
        # Do not pool non-leaf / views that own no storage safely — only
        # contiguous tensors allocated by us.
        if not t.is_contiguous():
            return
        k = _key(tuple(t.shape), t.dtype, t.device)
        bucket = self._free[k]
        if len(bucket) >= self.max_buffers_per_key:
            return
        self.releases += 1
        bucket.append(t)

    def clear(self) -> None:
        self._free.clear()

    def stats(self) -> dict:
        pooled = sum(len(v) for v in self._free.values())
        return {
            "hits": self.hits,
            "misses": self.misses,
            "releases": self.releases,
            "pooled_buffers": pooled,
            "keys": len(self._free),
        }


# Process-wide default pool for Orbit hot paths
GLOBAL_WORKSPACE = WorkspacePool()
