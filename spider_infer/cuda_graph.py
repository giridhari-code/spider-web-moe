"""CUDA Graph capture/replay for fixed-shape decode steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class DecodeGraphKey:
    batch_size: int
    # Cache length is *not* part of the graph inputs if we use paged attention
    # with fixed buffer addresses; only the token tensor shape matters for
    # the static graph. We still key on a length bucket for safety.
    length_bucket: int

    def __hash__(self):
        return hash((self.batch_size, self.length_bucket))


class CudaGraphDecodeRunner:
    """Capture and replay a decode callable for static shapes.

    The callable must accept (input_ids [B,1], request_ids) and return logits [B,1,V].
    Side effects on the paged KV cache are allowed because physical buffers are
    preallocated at fixed addresses.
    """

    def __init__(self, enabled: bool = True, length_bucket_size: int = 64):
        self.enabled = enabled and torch.cuda.is_available()
        self.length_bucket_size = length_bucket_size
        self._graphs: dict[DecodeGraphKey, torch.cuda.CUDAGraph] = {}
        self._static_ids: dict[DecodeGraphKey, torch.Tensor] = {}
        self._static_logits: dict[DecodeGraphKey, torch.Tensor] = {}
        self._static_rids: dict[DecodeGraphKey, list] = {}
        self._captured_fn: Callable | None = None

    def _bucket(self, length: int) -> int:
        b = self.length_bucket_size
        return ((length + b - 1) // b) * b

    def clear(self):
        self._graphs.clear()
        self._static_ids.clear()
        self._static_logits.clear()
        self._static_rids.clear()

    def run(
        self,
        fn: Callable,
        input_ids: torch.Tensor,
        request_ids: list,
        cache_length: int,
    ) -> torch.Tensor:
        """Run fn with optional CUDA graph replay.

        fn(input_ids, request_ids) -> logits [B, 1, V]
        """
        if (
            not self.enabled
            or input_ids.device.type != "cuda"
            or input_ids.shape[1] != 1
        ):
            return fn(input_ids, request_ids)

        key = DecodeGraphKey(
            batch_size=input_ids.shape[0],
            length_bucket=self._bucket(cache_length),
        )

        if key not in self._graphs:
            # Warmup (required before capture)
            for _ in range(3):
                logits = fn(input_ids, request_ids)
            torch.cuda.synchronize()

            static_ids = input_ids.clone()
            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_logits = fn(static_ids, request_ids)
            self._graphs[key] = g
            self._static_ids[key] = static_ids
            self._static_logits[key] = static_logits
            self._static_rids[key] = list(request_ids)
            return static_logits.clone()

        # Replay: copy inputs into static buffer
        # request_ids must match the captured set for page tables to be valid;
        # if they differ, fall back to eager.
        if list(request_ids) != self._static_rids[key]:
            return fn(input_ids, request_ids)

        self._static_ids[key].copy_(input_ids)
        self._graphs[key].replay()
        return self._static_logits[key].clone()
