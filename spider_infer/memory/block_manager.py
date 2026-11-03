"""High-level block manager facade over GlobalPagedKVCache.

Phase 1 keeps the existing GlobalPagedKVCache as the source of truth and
exposes a scheduler-friendly API (can_allocate, allocate, free, stats).
"""

from __future__ import annotations

from enum import Enum
from typing import Hashable

import torch

from spider_moe.global_paged_kv import GlobalPagedKVCache
from spider_moe.model import SpiderWebLM

from ..config import CacheConfig


class AllocStatus(Enum):
    OK = "ok"
    LATER = "later"   # temporarily out of pages / slots
    NEVER = "never"   # request exceeds hard limits


class BlockManager:
    """Owns the global paged KV pool and request → page mappings."""

    def __init__(
        self,
        model: SpiderWebLM,
        cache_config: CacheConfig,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.cfg = cache_config
        param = next(model.parameters())
        device = device or param.device
        dtype = cache_config.torch_dtype()

        num_layers = len(model._attention_layers())
        num_heads = model.cfg.num_heads
        head_dim = model.cfg.hidden_size // model.cfg.num_heads

        self.cache = GlobalPagedKVCache(
            num_layers=num_layers,
            num_pages=cache_config.num_pages,
            page_size=cache_config.page_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_requests=cache_config.max_requests,
            max_blocks_per_request=cache_config.max_blocks_per_request,
            dtype=dtype,
            device=device,
            fp8_scale_mode=getattr(cache_config, "fp8_scale_mode", "page"),
        )
        self.page_size = cache_config.page_size
        self.max_blocks_per_request = cache_config.max_blocks_per_request

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def pages_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.page_size - 1) // self.page_size

    def free_pages(self) -> int:
        return self.cache.stats().get("free_pages", 0)

    def free_slots(self) -> int:
        stats = self.cache.stats()
        active = stats.get("active_requests", 0)
        return max(0, self.cfg.max_requests - active)

    def usage_ratio(self) -> float:
        stats = self.cache.stats()
        total = stats.get("total_pages", 1)
        free = stats.get("free_pages", 0)
        return 1.0 - (free / max(total, 1))


    def can_allocate(self, num_tokens: int) -> AllocStatus:
        """Check whether a new request of `num_tokens` can be admitted."""
        needed = self.pages_for_tokens(num_tokens)
        if needed > self.max_blocks_per_request:
            return AllocStatus.NEVER
        if self.free_slots() < 1:
            return AllocStatus.LATER
        if self.free_pages() < needed:
            return AllocStatus.LATER
        return AllocStatus.OK

    def can_append(self, request_id: Hashable, extra_tokens: int = 1) -> AllocStatus:
        """Check whether extra tokens can be appended to an existing request."""
        try:
            current_len = self.cache.sequence_length(request_id)
        except KeyError:
            return AllocStatus.NEVER
        needed_pages = self.pages_for_tokens(current_len + extra_tokens)
        if needed_pages > self.max_blocks_per_request:
            return AllocStatus.NEVER
        # Already-owned pages do not need new allocation.
        current_pages = self.pages_for_tokens(current_len)
        extra_pages = max(0, needed_pages - current_pages)
        if extra_pages > self.free_pages():
            return AllocStatus.LATER
        return AllocStatus.OK

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate(self, request_id: Hashable, num_tokens: int = 0) -> None:
        status = self.can_allocate(max(num_tokens, 1))
        if status is AllocStatus.NEVER:
            raise RuntimeError(
                f"request {request_id} needs {num_tokens} tokens which exceeds "
                f"max_blocks_per_request={self.max_blocks_per_request}"
            )
        if status is AllocStatus.LATER:
            raise RuntimeError(
                f"cannot allocate pages for {request_id} right now "
                f"(free_pages={self.free_pages()}, free_slots={self.free_slots()})"
            )
        self.cache.allocate(request_id, seq_len=num_tokens)

    def reserve(self, request_id: Hashable, required_length: int) -> None:
        self.cache.reserve(request_id, required_length)

    def free(self, request_id: Hashable) -> None:
        try:
            self.cache.free(request_id)
        except KeyError:
            pass

    def fork_prefix(
        self,
        source_id: Hashable,
        new_id: Hashable,
        prefix_len: int | None = None,
    ) -> None:
        """Share source request's KV pages with a new request (refcount + COW)."""
        if self.free_slots() < 1:
            raise RuntimeError("no free request slots for fork_prefix")
        self.cache.fork_prefix(source_id, new_id, prefix_len=prefix_len)

    def has_request(self, request_id: Hashable) -> bool:
        try:
            self.cache.sequence_length(request_id)
            return True
        except KeyError:
            return False

    def sequence_length(self, request_id: Hashable) -> int:
        return self.cache.sequence_length(request_id)

    def stats(self) -> dict:
        return self.cache.stats()
