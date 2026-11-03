"""CUDA Graph capture/replay for fixed-shape decode steps (Orbit 0.3).

Graph identity is (batch_size, length_bucket, num_loops).
early_exit must be False at capture/replay time — ActiveSet size is dynamic
under early exit and cannot be fused into a static graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DecodeGraphKey:
    """Orbit CUDA graph key: active batch A, KV length bucket, fixed loop count L."""

    batch_size: int
    length_bucket: int
    num_loops: int = 1

    def __hash__(self):
        return hash((self.batch_size, self.length_bucket, self.num_loops))


class CudaGraphDecodeRunner:
    """Capture and replay a decode callable for static shapes.

    The callable must accept (input_ids [B,1], request_ids) and return logits [B,1,V].
    Side effects on the paged KV cache are allowed because physical buffers are
    preallocated at fixed addresses.

    Orbit 0.3: key includes num_loops so different orbit depths do not collide.
    """

    def __init__(self, enabled: bool = True, length_bucket_size: int = 64):
        self.enabled = enabled and torch.cuda.is_available()
        self.length_bucket_size = length_bucket_size
        self._graphs: dict[DecodeGraphKey, torch.cuda.CUDAGraph] = {}
        self._static_ids: dict[DecodeGraphKey, torch.Tensor] = {}
        self._static_logits: dict[DecodeGraphKey, torch.Tensor] = {}
        self._static_rids: dict[DecodeGraphKey, list] = {}
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0

    def _bucket(self, length: int) -> int:
        b = self.length_bucket_size
        return ((length + b - 1) // b) * b

    def clear(self):
        self._graphs.clear()
        self._static_ids.clear()
        self._static_logits.clear()
        self._static_rids.clear()
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "graphs": len(self._graphs),
            "hits": self.hits,
            "misses": self.misses,
            "fallbacks": self.fallbacks,
            "keys": [
                {
                    "batch_size": k.batch_size,
                    "length_bucket": k.length_bucket,
                    "num_loops": k.num_loops,
                }
                for k in self._graphs
            ],
        }

    def run(
        self,
        fn: Callable,
        input_ids: torch.Tensor,
        request_ids: list,
        cache_length: int,
        num_loops: int = 1,
    ) -> torch.Tensor:
        """Run fn with optional CUDA graph replay.

        fn(input_ids, request_ids) -> logits [B, 1, V]
        """
        if (
            not self.enabled
            or input_ids.device.type != "cuda"
            or input_ids.shape[1] != 1
        ):
            self.fallbacks += 1
            return fn(input_ids, request_ids)

        key = DecodeGraphKey(
            batch_size=input_ids.shape[0],
            length_bucket=self._bucket(cache_length),
            num_loops=max(1, int(num_loops)),
        )

        if key not in self._graphs:
            self.misses += 1
            # Warmup (required before capture)
            for _ in range(3):
                logits = fn(input_ids, request_ids)
            torch.cuda.synchronize()

            static_ids = input_ids.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_logits = fn(static_ids, request_ids)
            self._graphs[key] = g
            self._static_ids[key] = static_ids
            self._static_logits[key] = static_logits
            self._static_rids[key] = list(request_ids)
            return static_logits.clone()

        # Replay: request_ids must match captured set for page tables.
        if list(request_ids) != self._static_rids[key]:
            self.fallbacks += 1
            return fn(input_ids, request_ids)

        self.hits += 1
        self._static_ids[key].copy_(input_ids)
        self._graphs[key].replay()
        return self._static_logits[key].clone()
