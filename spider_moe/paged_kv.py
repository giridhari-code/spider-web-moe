"""Loop-aware paged KV cache.

Pools are isolated per (layer, loop), matching the inference design's
separate per-layer/per-loop cache ownership. Tensors are allocated on the
requested device; block tables are exposed as device tensors for indirect
addressing by future CUDA graph kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable
import math
import torch


@dataclass
class _RequestState:
    pages: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    lengths: dict[tuple[int, int], int] = field(default_factory=dict)


class _PagePool:
    def __init__(self, num_pages: int, page_size: int, num_heads: int,
                 head_dim: int, dtype: torch.dtype, device: torch.device):
        if num_pages < 1:
            raise ValueError("num_pages must be >= 1")
        self.page_size = page_size
        self.keys = torch.empty((num_pages, page_size, num_heads, head_dim),
                                dtype=dtype, device=device)
        self.values = torch.empty_like(self.keys)
        self.free_pages = list(range(num_pages - 1, -1, -1))
        self.num_pages = num_pages

    def take(self) -> int:
        if not self.free_pages:
            raise RuntimeError("Paged KV cache is out of physical pages")
        return self.free_pages.pop()

    def release(self, page_ids: list[int]):
        for page in page_ids:
            if page < 0 or page >= self.num_pages:
                raise ValueError(f"invalid physical page id: {page}")
            self.free_pages.append(page)


class PagedKVCache:
    """Paged cache with an independent physical page pool per layer and loop.

    K/V tensor shape per pool:
        [num_physical_pages, page_size, num_heads, head_dim]

    Logical-to-physical mapping:
        block_table[request_id][layer][loop][logical_block] = physical_page_id

    Page size defaults to 16, as in INFERENCE.md. Use append() with tensors
    shaped [tokens, heads, head_dim]. The cache stores contiguous token order
    and gather() returns [sequence, heads, head_dim].
    """
    def __init__(self, num_layers: int, num_loops: int, num_pages: int,
                 page_size: int = 16, num_heads: int = 8, head_dim: int = 64,
                 dtype: torch.dtype = torch.float16, device: str | torch.device = "cpu",
                 max_blocks_per_request: int | None = None):
        if min(num_layers, num_loops, page_size, num_heads, head_dim) <= 0:
            raise ValueError("all dimensions and page_size must be positive")
        self.num_layers, self.num_loops = num_layers, num_loops
        self.page_size = page_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_blocks_per_request = max_blocks_per_request
        self.pools = {
            (layer, loop): _PagePool(num_pages, page_size, num_heads, head_dim,
                                     dtype, self.device)
            for layer in range(num_layers) for loop in range(num_loops)
        }
        self.requests: dict[Hashable, _RequestState] = {}

    def _validate_key(self, layer: int, loop: int):
        if not 0 <= layer < self.num_layers:
            raise IndexError("layer out of range")
        if not 0 <= loop < self.num_loops:
            raise IndexError("loop out of range")

    def _state(self, request_id: Hashable) -> _RequestState:
        if request_id not in self.requests:
            self.requests[request_id] = _RequestState()
        return self.requests[request_id]

    def allocate(self, request_id: Hashable, seq_len: int = 0,
                 layer: int | None = None, loop: int | None = None):
        """Reserve enough pages for seq_len; omit layer/loop to reserve all pools."""
        if seq_len < 0:
            raise ValueError("seq_len cannot be negative")
        keys = ([(layer, loop)] if layer is not None and loop is not None
                else list(self.pools))
        state = self._state(request_id)
        allocated = []
        try:
            for key in keys:
                self._validate_key(*key)
                pool = self.pools[key]
                pages = state.pages.setdefault(key, [])
                needed = math.ceil(seq_len / self.page_size)
                if self.max_blocks_per_request is not None and needed > self.max_blocks_per_request:
                    raise RuntimeError("request exceeds max_blocks_per_request")
                while len(pages) < needed:
                    pages.append(pool.take())
                    allocated.append((pool, pages))
        except Exception:
            # Roll back only pages acquired by this call.
            for pool, pages in reversed(allocated):
                page = pages.pop()
                pool.release([page])
            raise
        return {key: list(state.pages.get(key, [])) for key in keys}

    def append(self, request_id: Hashable, layer: int, loop: int,
               keys: torch.Tensor, values: torch.Tensor):
        """Append K/V tokens. Input shape [T, heads, head_dim]."""
        self._validate_key(layer, loop)
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("keys and values must have equal [T, heads, head_dim] shapes")
        if keys.dtype != self.dtype or values.dtype != self.dtype:
            keys, values = keys.to(self.dtype), values.to(self.dtype)
        keys, values = keys.to(self.device), values.to(self.device)
        state = self._state(request_id)
        key = (layer, loop)
        pages = state.pages.setdefault(key, [])
        length = state.lengths.get(key, 0)
        new_length = length + keys.shape[0]
        needed = math.ceil(new_length / self.page_size)
        if self.max_blocks_per_request is not None and needed > self.max_blocks_per_request:
            raise RuntimeError("request exceeds max_blocks_per_request")
        pool = self.pools[key]
        allocated = []
        try:
            while len(pages) < needed:
                page = pool.take()
                pages.append(page)
                allocated.append(page)
        except Exception:
            for page in reversed(allocated):
                pages.pop()
                pool.release([page])
            raise
        for offset in range(keys.shape[0]):
            pos = length + offset
            logical_block, in_page = divmod(pos, self.page_size)
            physical_page = pages[logical_block]
            pool.keys[physical_page, in_page].copy_(keys[offset])
            pool.values[physical_page, in_page].copy_(values[offset])
        state.lengths[key] = new_length
        return new_length

    def gather(self, request_id: Hashable, layer: int, loop: int):
        """Materialize this request's logical K/V sequence in token order."""
        self._validate_key(layer, loop)
        if request_id not in self.requests:
            raise KeyError(f"unknown request: {request_id}")
        state = self.requests[request_id]
        key = (layer, loop)
        length = state.lengths.get(key, 0)
        pages = state.pages.get(key, [])
        if length == 0:
            pool = self.pools[key]
            shape = (0, pool.keys.shape[2], pool.keys.shape[3])
            return (torch.empty(shape, dtype=self.dtype, device=self.device),
                    torch.empty(shape, dtype=self.dtype, device=self.device))
        pool = self.pools[key]
        k_parts, v_parts = [], []
        remaining = length
        for page in pages:
            take = min(self.page_size, remaining)
            k_parts.append(pool.keys[page, :take])
            v_parts.append(pool.values[page, :take])
            remaining -= take
            if remaining == 0:
                break
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def block_table(self, request_id: Hashable, layer: int, loop: int,
                    max_blocks: int | None = None):
        """Return device-resident [max_blocks] int32 table, padded with -1."""
        self._validate_key(layer, loop)
        if request_id not in self.requests:
            raise KeyError(f"unknown request: {request_id}")
        pages = self.requests[request_id].pages.get((layer, loop), [])
        width = max_blocks or self.max_blocks_per_request or max(1, len(pages))
        if len(pages) > width:
            raise ValueError("max_blocks is smaller than allocated block count")
        table = torch.full((width,), -1, dtype=torch.int32, device=self.device)
        if pages:
            table[:len(pages)] = torch.tensor(pages, dtype=torch.int32, device=self.device)
        return table

    def sequence_length(self, request_id: Hashable, layer: int, loop: int) -> int:
        self._validate_key(layer, loop)
        if request_id not in self.requests:
            return 0
        return self.requests[request_id].lengths.get((layer, loop), 0)

    def free(self, request_id: Hashable, layer: int | None = None,
             loop: int | None = None):
        """Release request pages; can release one layer/loop pool or all pools."""
        if request_id not in self.requests:
            return
        state = self.requests[request_id]
        if layer is None and loop is None:
            keys = list(state.pages)
        elif layer is not None and loop is not None:
            self._validate_key(layer, loop)
            keys = [(layer, loop)]
        else:
            raise ValueError("specify both layer and loop, or neither")
        for key in keys:
            self.pools[key].release(state.pages.pop(key, []))
            state.lengths.pop(key, None)
        if not state.pages:
            self.requests.pop(request_id, None)

    def free_page_count(self, layer: int, loop: int) -> int:
        self._validate_key(layer, loop)
        return len(self.pools[(layer, loop)].free_pages)

    def stats(self):
        total = sum(pool.num_pages for pool in self.pools.values())
        free = sum(len(pool.free_pages) for pool in self.pools.values())
        return {"total_physical_pages": total, "free_physical_pages": free,
                "used_physical_pages": total-free, "page_size": self.page_size,
                "pool_count": len(self.pools), "active_requests": len(self.requests)}
