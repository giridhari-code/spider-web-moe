"""Paged attention backends.

- PyTorch reference backend: portable, correctness-oriented, gathers pages.
- Optional Triton CUDA decode kernel: reads physical pages directly for T=1.

The Triton path is deliberately limited to single-token decode. Prefill and
multi-token chunks use the reference backend until a tiled paged-prefill kernel
is added and benchmarked.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .quant import is_fp8_dtype

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

LAST_PAGED_ATTENTION_BACKEND = "not_run"
LAST_PAGED_ATTENTION_ERROR = None


if TRITON_AVAILABLE:
    @triton.jit
    def _paged_decode_kernel(
        Q, K, V, BLOCK_TABLE, SEQ_LENS, REQUEST_SLOTS, OUT,
        q_s0: tl.constexpr, q_s1: tl.constexpr, q_s2: tl.constexpr,
        k_s0: tl.constexpr, k_s1: tl.constexpr, k_s2: tl.constexpr,
        k_s3: tl.constexpr, k_s4: tl.constexpr,
        v_s0: tl.constexpr, v_s1: tl.constexpr, v_s2: tl.constexpr,
        v_s3: tl.constexpr, v_s4: tl.constexpr,
        bt_s0: tl.constexpr, bt_s1: tl.constexpr,
        out_s0: tl.constexpr, out_s1: tl.constexpr, out_s2: tl.constexpr,
        NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        LAYER: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_id = pid // NUM_HEADS
        head_id = pid % NUM_HEADS
        request_slot = tl.load(REQUEST_SLOTS + batch_id)
        seq_len = tl.load(SEQ_LENS + request_slot)
        d = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q + batch_id*q_s0 + head_id*q_s1 + d*q_s2,
            d < HEAD_DIM, other=0.0
        ).to(tl.float32)

        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        running_acc = tl.full((BLOCK_D,), 0.0, tl.float32)

        num_blocks = tl.cdiv(seq_len, PAGE_SIZE)
        for logical_block in range(num_blocks):
            physical_page = tl.load(
                BLOCK_TABLE + request_slot*bt_s0 + logical_block*bt_s1
            )
            for page_offset in tl.static_range(0, PAGE_SIZE):
                token_pos = logical_block*PAGE_SIZE + page_offset
                if token_pos < seq_len:
                    k = tl.load(
                        K + physical_page*k_s0 + LAYER*k_s1
                          + page_offset*k_s2 + head_id*k_s3 + d*k_s4,
                        d < HEAD_DIM, other=0.0
                    ).to(tl.float32)
                    v = tl.load(
                        V + physical_page*v_s0 + LAYER*v_s1
                          + page_offset*v_s2 + head_id*v_s3 + d*v_s4,
                        d < HEAD_DIM, other=0.0
                    ).to(tl.float32)
                    score = tl.sum(q * k, axis=0) * SCALE
                    new_max = tl.maximum(running_max, score)
                    old_scale = tl.exp(running_max - new_max)
                    new_scale = tl.exp(score - new_max)
                    running_acc = running_acc * old_scale + v * new_scale
                    running_sum = running_sum * old_scale + new_scale
                    running_max = new_max

        result = running_acc / running_sum
        tl.store(
            OUT + batch_id*out_s0 + head_id*out_s1 + d*out_s2,
            result, d < HEAD_DIM
        )


def _reference_paged_attention(q, cache, request_ids, layer, query_start):
    """Portable SDPA fallback; materializes each request's K/V prefix."""
    outputs = []
    query_len = q.shape[2]
    for batch_index, request_id in enumerate(request_ids):
        k, v = cache.read_layer(request_id, layer)
        k = k.transpose(0, 1).unsqueeze(0).to(dtype=q.dtype)
        v = v.transpose(0, 1).unsqueeze(0).to(dtype=q.dtype)
        q_row = q[batch_index:batch_index + 1]
        if query_start == 0:
            y = F.scaled_dot_product_attention(q_row, k, v, is_causal=True)
        elif query_len == 1:
            y = F.scaled_dot_product_attention(q_row, k, v, is_causal=False)
        else:
            q_pos = query_start + torch.arange(query_len, device=q.device)
            k_pos = torch.arange(k.shape[2], device=q.device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            y = F.scaled_dot_product_attention(q_row, k, v, attn_mask=mask)
        outputs.append(y)
    return torch.cat(outputs, dim=0)


def _triton_single_token(q, cache, request_ids, layer):
    batch, heads, query_len, head_dim = q.shape
    if query_len != 1:
        return None
    if head_dim > 128:
        return None
    request_slots = cache.batch_request_slots(request_ids)
    out = torch.empty_like(q)
    q3 = q.contiguous().squeeze(2)
    out3 = out.squeeze(2)
    _paged_decode_kernel[(batch * heads,)](
        q3, cache.keys, cache.values,
        cache.block_tables, cache.sequence_lengths, request_slots,
        out3,
        q3.stride(0), q3.stride(1), q3.stride(2),
        cache.keys.stride(0), cache.keys.stride(1), cache.keys.stride(2),
        cache.keys.stride(3), cache.keys.stride(4),
        cache.values.stride(0), cache.values.stride(1), cache.values.stride(2),
        cache.values.stride(3), cache.values.stride(4),
        cache.block_tables.stride(0), cache.block_tables.stride(1),
        out3.stride(0), out3.stride(1), out3.stride(2),
        heads, head_dim, cache.page_size, cache.max_blocks_per_request,
        layer, 1.0 / math.sqrt(head_dim),
        triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out


def paged_attention(q, cache, request_ids, layer, query_start):
    """Compute attention over block-table-addressed K/V.

    Args:
      q: [batch, heads, query_tokens, head_dim]
      request_ids: one cache request ID per batch row.
      query_start: logical position of the first query token.
    """
    if q.ndim != 4:
        raise ValueError("q must have shape [B,H,T,D]")
    if len(request_ids) != q.shape[0]:
        raise ValueError("request_ids length must equal query batch size")
    if cache.device != q.device:
        raise ValueError("query and cache must be on the same device")
    if q.shape[1] != cache.num_heads or q.shape[3] != cache.head_dim:
        raise ValueError("query head dimensions do not match paged cache")
    global LAST_PAGED_ATTENTION_BACKEND, LAST_PAGED_ATTENTION_ERROR
    LAST_PAGED_ATTENTION_ERROR = None
    # Triton prototype assumes FP16/BF16 storage; FP8 uses the reference path.
    if (
        TRITON_AVAILABLE and q.is_cuda and q.shape[2] == 1
        and not is_fp8_dtype(cache.dtype)
    ):
        try:
            result = _triton_single_token(q, cache, request_ids, layer)
            if result is not None:
                LAST_PAGED_ATTENTION_BACKEND = "triton_cuda_decode"
                return result
        except Exception as exc:
            # Keep the reference path available if a particular Triton build,
            # GPU architecture, or shape cannot compile this prototype kernel.
            LAST_PAGED_ATTENTION_ERROR = repr(exc)
    LAST_PAGED_ATTENTION_BACKEND = "torch_reference"
    return _reference_paged_attention(q, cache, request_ids, layer, query_start)


def get_last_paged_attention_backend():
    """Return the backend used by the most recent paged attention invocation."""
    return LAST_PAGED_ATTENTION_BACKEND, LAST_PAGED_ATTENTION_ERROR
