"""Global block-pool paged KV cache with stable device metadata.

One physical block stores every attention layer's K and V for a token range.
Thus a request owns one logical block table shared by all attention layers,
rather than duplicating a full page pool for each layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import torch

from .quant import (
    dequantize_fp8,
    dequantize_fp8_with_scale_tensor,
    is_fp8_dtype,
    quantize_fp8,
    quantize_fp8_with_scale_tensor,
)


@dataclass
class _Request:
    slot: int
    pages: list[int] = field(default_factory=list)
    length: int = 0


class GlobalPagedKVCache:
    """Global physical page pool shared by all requests and attention layers.

    Storage layout:
      K,V: [physical_pages, layers, page_size, heads, head_dim]

    Block table:
      [max_requests, max_blocks], int32 on the cache device.
    Sequence lengths:
      [max_requests], int32 on the cache device.

    `num_pages` is the total physical page budget, not a per-layer budget.
    """
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        max_requests: int = 32,
        max_blocks_per_request: int = 256,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cpu",
        fp8_scale_mode: str = "page",
    ):
        dims = (num_layers, num_pages, page_size, num_heads, head_dim,
                max_requests, max_blocks_per_request)
        if any(v <= 0 for v in dims):
            raise ValueError("all cache dimensions must be positive")
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_requests = max_requests
        self.max_blocks_per_request = max_blocks_per_request
        self.dtype = dtype
        self.device = torch.device(device)
        # Attention matmuls run in this dtype when storage is FP8.
        self.compute_dtype = (
            torch.float16 if is_fp8_dtype(dtype) else dtype
        )

        shape = (num_pages, num_layers, page_size, num_heads, head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device=self.device)
        self.values = torch.empty_like(self.keys)
        # FP8 scale tensors. Shape depends on fp8_scale_mode:
        #   page       — [P, L]
        #   head       — [P, L, H]
        #   token      — [P, L, S]
        #   token_head — [P, L, S, H]
        self.fp8_scale_mode = fp8_scale_mode
        if is_fp8_dtype(dtype):
            mode = fp8_scale_mode
            if mode not in ("page", "head", "token", "token_head"):
                raise ValueError(
                    f"fp8_scale_mode must be page|head|token|token_head, got {mode}"
                )
            if mode == "page":
                scale_shape = (num_pages, num_layers)
            elif mode == "head":
                scale_shape = (num_pages, num_layers, num_heads)
            elif mode == "token":
                scale_shape = (num_pages, num_layers, page_size)
            else:  # token_head
                scale_shape = (num_pages, num_layers, page_size, num_heads)
            self.key_scales = torch.ones(
                scale_shape, dtype=torch.float32, device=self.device
            )
            self.value_scales = torch.ones_like(self.key_scales)
        else:
            self.key_scales = None
            self.value_scales = None

        # Fixed-address metadata buffers are suitable for later graph replay.
        self.block_tables = torch.full(
            (max_requests, max_blocks_per_request), -1,
            dtype=torch.int32, device=self.device,
        )
        self.sequence_lengths = torch.zeros(
            (max_requests,), dtype=torch.int32, device=self.device,
        )
        self.request_slots = torch.arange(
            max_requests, dtype=torch.int32, device=self.device,
        )
        self._batch_slots = torch.empty(
            (max_requests,), dtype=torch.int32, device=self.device,
        )

        self._free_pages = list(range(num_pages - 1, -1, -1))
        self._page_refcounts = [0] * num_pages
        self._free_slots = list(range(max_requests - 1, -1, -1))
        self._requests: dict[Hashable, _Request] = {}

    def _request(self, request_id: Hashable) -> _Request:
        try:
            return self._requests[request_id]
        except KeyError:
            raise KeyError(f"unknown request id: {request_id}") from None

    def _new_request(self, request_id: Hashable) -> _Request:
        if request_id in self._requests:
            raise ValueError(f"request already exists: {request_id}")
        if not self._free_slots:
            raise RuntimeError("maximum active request slots reached")
        slot = self._free_slots.pop()
        req = _Request(slot=slot)
        self._requests[request_id] = req
        self.block_tables[slot].fill_(-1)
        self.sequence_lengths[slot] = 0
        return req

    def _alloc_page(self) -> int:
        if not self._free_pages:
            raise RuntimeError("global paged KV pool is out of physical pages")
        page = self._free_pages.pop()
        self._page_refcounts[page] = 1
        if self.key_scales is not None:
            self.key_scales[page].fill_(1.0)
            self.value_scales[page].fill_(1.0)
        return page

    def _incref(self, page: int):
        if self._page_refcounts[page] <= 0:
            raise RuntimeError("cannot reference a free physical page")
        self._page_refcounts[page] += 1

    def _decref(self, page: int):
        self._page_refcounts[page] -= 1
        if self._page_refcounts[page] < 0:
            raise RuntimeError("physical page reference count underflow")
        if self._page_refcounts[page] == 0:
            self._free_pages.append(page)

    def _sync_table(self, req: _Request):
        self.block_tables[req.slot].fill_(-1)
        if req.pages:
            self.block_tables[req.slot, :len(req.pages)] = torch.tensor(
                req.pages, dtype=torch.int32, device=self.device
            )
        self.sequence_lengths[req.slot] = req.length

    def _ensure_capacity(self, req: _Request, required_length: int):
        if required_length < 0:
            raise ValueError("required_length cannot be negative")
        needed = (required_length + self.page_size - 1) // self.page_size
        if needed > self.max_blocks_per_request:
            raise RuntimeError("request exceeds max_blocks_per_request")
        allocated = []
        try:
            while len(req.pages) < needed:
                page = self._alloc_page()
                req.pages.append(page)
                allocated.append(page)
                self._sync_table(req)
        except Exception:
            for page in reversed(allocated):
                req.pages.pop()
                self._decref(page)
            self._sync_table(req)
            raise

    def _ensure_writable(self, req: _Request, logical_block: int):
        page = req.pages[logical_block]
        if self._page_refcounts[page] == 1:
            return page
        # Copy-on-write preserves shared prefix pages for forked requests.
        new_page = self._alloc_page()
        self.keys[new_page].copy_(self.keys[page])
        self.values[new_page].copy_(self.values[page])
        if self.key_scales is not None:
            self.key_scales[new_page].copy_(self.key_scales[page])
            self.value_scales[new_page].copy_(self.value_scales[page])
        req.pages[logical_block] = new_page
        self._decref(page)
        self._sync_table(req)
        return new_page

    def allocate(self, request_id: Hashable, seq_len: int = 0):
        """Create request and reserve enough global physical pages."""
        if seq_len < 0:
            raise ValueError("seq_len cannot be negative")
        req = self._new_request(request_id)
        try:
            self._ensure_capacity(req, seq_len)
            # Reservation does not mean K/V data has been written yet.
            req.length = 0
            self._sync_table(req)
        except Exception:
            self.free(request_id)
            raise
        return self.block_table(request_id)

    def reserve(self, request_id: Hashable, required_length: int):
        """Reserve enough pages for required logical sequence length."""
        req = self._request(request_id)
        self._ensure_capacity(req, required_length)
        return self.block_table(request_id)


    def _write_fp8_token(
        self,
        page: int,
        layer: int,
        offset: int,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
    ) -> None:
        """Quantize one token [H, D] into the page with mode-specific scales."""
        k_tok = k_tok.float()
        v_tok = v_tok.float()
        mode = self.fp8_scale_mode
        eps = 1e-12
        max_f = 448.0

        if mode == "page":
            for store, tok, scales in (
                (self.keys, k_tok, self.key_scales),
                (self.values, v_tok, self.value_scales),
            ):
                new_s = (tok.abs().amax() / max_f).clamp_min(eps)
                old_s = scales[page, layer]
                if new_s > old_s:
                    filled = offset
                    if filled > 0:
                        prev = dequantize_fp8(
                            store[page, layer, :filled], old_s, torch.float32
                        )
                        scales[page, layer] = new_s
                        q_prev, _ = quantize_fp8(prev, scale=new_s)
                        store[page, layer, :filled].copy_(q_prev)
                    else:
                        scales[page, layer] = new_s
                q, _ = quantize_fp8(tok, scale=scales[page, layer])
                store[page, layer, offset].copy_(q)

        elif mode == "head":
            for store, tok, scales in (
                (self.keys, k_tok, self.key_scales),
                (self.values, v_tok, self.value_scales),
            ):
                new_s = (tok.abs().amax(dim=-1) / max_f).clamp_min(eps)
                old_s = scales[page, layer]
                if (new_s > old_s).any() and offset > 0:
                    prev = dequantize_fp8_with_scale_tensor(
                        store[page, layer, :offset],
                        old_s.view(1, -1, 1),
                        torch.float32,
                    )
                    scales[page, layer] = torch.maximum(old_s, new_s)
                    q_prev = quantize_fp8_with_scale_tensor(
                        prev, scales[page, layer].view(1, -1, 1)
                    )
                    store[page, layer, :offset].copy_(q_prev)
                else:
                    scales[page, layer] = torch.maximum(old_s, new_s)
                q = quantize_fp8_with_scale_tensor(
                    tok, scales[page, layer].view(-1, 1)
                )
                store[page, layer, offset].copy_(q)

        elif mode == "token":
            for store, tok, scales in (
                (self.keys, k_tok, self.key_scales),
                (self.values, v_tok, self.value_scales),
            ):
                s = (tok.abs().amax() / max_f).clamp_min(eps)
                scales[page, layer, offset] = s
                q, _ = quantize_fp8(tok, scale=s)
                store[page, layer, offset].copy_(q)

        else:  # token_head
            for store, tok, scales in (
                (self.keys, k_tok, self.key_scales),
                (self.values, v_tok, self.value_scales),
            ):
                s = (tok.abs().amax(dim=-1) / max_f).clamp_min(eps)
                scales[page, layer, offset] = s
                q = quantize_fp8_with_scale_tensor(tok, s.view(-1, 1))
                store[page, layer, offset].copy_(q)

    def _dequant_fp8_tile(
        self,
        tile: torch.Tensor,
        scales: torch.Tensor,
        page: int,
        layer: int,
        take: int,
    ) -> torch.Tensor:
        """Dequantize a tile [take, H, D] using the configured scale mode."""
        mode = self.fp8_scale_mode
        if mode == "page":
            return dequantize_fp8(tile, scales[page, layer], self.compute_dtype)
        if mode == "head":
            return dequantize_fp8_with_scale_tensor(
                tile, scales[page, layer].view(1, -1, 1), self.compute_dtype
            )
        if mode == "token":
            return dequantize_fp8_with_scale_tensor(
                tile, scales[page, layer, :take].view(-1, 1, 1), self.compute_dtype
            )
        return dequantize_fp8_with_scale_tensor(
            tile, scales[page, layer, :take].unsqueeze(-1), self.compute_dtype
        )

    def write_layer(
        self,
        request_id: Hashable,
        layer: int,
        start_pos: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Write one layer's K/V for a token span [start_pos, start_pos+T)."""
        if not 0 <= layer < self.num_layers:
            raise IndexError("layer out of range")
        if start_pos < 0:
            raise ValueError("start_pos cannot be negative")
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("K/V must have equal [tokens, heads, head_dim] shapes")
        if keys.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("K/V head shape does not match cache configuration")
        req = self._request(request_id)
        end = start_pos + keys.shape[0]
        self._ensure_capacity(req, end)
        keys = keys.to(device=self.device)
        values = values.to(device=self.device)
        use_fp8 = is_fp8_dtype(self.dtype)
        if not use_fp8:
            keys = keys.to(dtype=self.dtype)
            values = values.to(dtype=self.dtype)
        for i in range(keys.shape[0]):
            pos = start_pos + i
            logical_block, offset = divmod(pos, self.page_size)
            page = self._ensure_writable(req, logical_block)
            if use_fp8:
                self._write_fp8_token(page, layer, offset, keys[i], values[i])
            else:
                self.keys[page, layer, offset].copy_(keys[i])
                self.values[page, layer, offset].copy_(values[i])
        req.length = max(req.length, end)
        self._sync_table(req)

    def write_layer_batch(
        self,
        request_ids: list[Hashable],
        layer: int,
        start_pos: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Batch wrapper. Input K/V [B,T,H,D]."""
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("batched K/V must have equal [B,T,H,D] shapes")
        if len(request_ids) != keys.shape[0]:
            raise ValueError("request_ids length must equal batch size")
        for b, request_id in enumerate(request_ids):
            self.write_layer(request_id, layer, start_pos, keys[b], values[b])

    def read_layer(self, request_id: Hashable, layer: int, upto: int | None = None):
        """Materialize a request's layer K/V in logical token order (reference path)."""
        if not 0 <= layer < self.num_layers:
            raise IndexError("layer out of range")
        req = self._request(request_id)
        length = req.length if upto is None else upto
        if not 0 <= length <= req.length:
            raise ValueError("requested length is outside the written sequence")
        out_dtype = self.compute_dtype if is_fp8_dtype(self.dtype) else self.dtype
        if length == 0:
            shape = (0, self.num_heads, self.head_dim)
            return (torch.empty(shape, dtype=out_dtype, device=self.device),
                    torch.empty(shape, dtype=out_dtype, device=self.device))
        k_parts, v_parts = [], []
        remaining = length
        for page in req.pages:
            take = min(self.page_size, remaining)
            k_tile = self.keys[page, layer, :take]
            v_tile = self.values[page, layer, :take]
            if is_fp8_dtype(self.dtype):
                k_tile = self._dequant_fp8_tile(
                    k_tile, self.key_scales, page, layer, take
                )
                v_tile = self._dequant_fp8_tile(
                    v_tile, self.value_scales, page, layer, take
                )
            k_parts.append(k_tile)
            v_parts.append(v_tile)
            remaining -= take
            if remaining == 0:
                break
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def block_table(self, request_id: Hashable):
        """Return a stable view into the preallocated device block-table row."""
        req = self._request(request_id)
        return self.block_tables[req.slot]

    def request_slot(self, request_id: Hashable) -> int:
        return self._request(request_id).slot

    def batch_request_slots(self, request_ids: list[Hashable]):
        """Fill and return a stable device buffer containing request slot IDs."""
        if len(request_ids) > self.max_requests:
            raise ValueError("batch exceeds max_requests")
        slots = [self._request(rid).slot for rid in request_ids]
        self._batch_slots[:len(slots)] = torch.tensor(
            slots, dtype=torch.int32, device=self.device
        )
        return self._batch_slots[:len(slots)]

    def sequence_length(self, request_id: Hashable) -> int:
        return self._request(request_id).length

    def fork_prefix(self, source_id: Hashable, new_id: Hashable, prefix_len: int | None = None):
        """Share complete/partial prefix pages; subsequent writes use copy-on-write."""
        src = self._request(source_id)
        if new_id in self._requests:
            raise ValueError(f"request already exists: {new_id}")
        length = src.length if prefix_len is None else prefix_len
        if not 0 <= length <= src.length:
            raise ValueError("prefix_len must be within source sequence")
        dst = self._new_request(new_id)
        needed = (length + self.page_size - 1) // self.page_size
        dst.pages = list(src.pages[:needed])
        for page in dst.pages:
            self._incref(page)
        dst.length = length
        self._sync_table(dst)
        return self.block_table(new_id)

    def trim_to_length(self, request_id: Hashable, length: int) -> int:
        """Drop physical pages beyond those needed for ``length`` tokens.

        Used after decode completes when only a prompt prefix must stay resident
        (or to release over-reserved capacity). Shared pages are refcounted.
        Returns the number of pages released.
        """
        if length < 0:
            raise ValueError("length cannot be negative")
        req = self._request(request_id)
        if length > req.length:
            # Cannot invent tokens; only shrink or no-op.
            length = req.length
        needed = (length + self.page_size - 1) // self.page_size if length > 0 else 0
        released = 0
        while len(req.pages) > needed:
            page = req.pages.pop()
            self._decref(page)
            released += 1
        req.length = length
        self._sync_table(req)
        return released

    def free(self, request_id: Hashable):
        """Release request's pages and request slot; shared pages are refcounted."""
        if request_id not in self._requests:
            return
        req = self._requests.pop(request_id)
        for page in req.pages:
            self._decref(page)
        self.block_tables[req.slot].fill_(-1)
        self.sequence_lengths[req.slot] = 0
        self._free_slots.append(req.slot)

    def stats(self):
        used = self.num_pages - len(self._free_pages)
        return {
            "total_pages": self.num_pages,
            "used_pages": used,
            "free_pages": len(self._free_pages),
            "active_requests": len(self._requests),
            "page_size": self.page_size,
            "shared_pages": sum(1 for count in self._page_refcounts if count > 1),
            "storage_bytes": self.keys.numel() * self.keys.element_size()
                            + self.values.numel() * self.values.element_size(),
        }
