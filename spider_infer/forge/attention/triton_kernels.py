"""Forge 0.2 — Triton paged decode attention kernels (FlashMLA-shaped).

Falls back gracefully when Triton or CUDA is missing. Layout for the dense
paged path used by ForgeAttention.paged_decode_flat:

  q:          [B, H, 1, D]
  k_cache:    [P, H, PAGE, D]   physical pages
  v_cache:    [P, H, PAGE, D]
  block_table:[B, max_blocks]   int32 page ids (-1 empty)
  seq_lens:   [B]               valid tokens per row
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

LAST_FORGE_ATTN_BACKEND = "not_run"
LAST_FORGE_ATTN_ERROR: str | None = None


if TRITON_AVAILABLE:

    @triton.jit
    def _forge_paged_decode_kernel(
        Q, K, V, BLOCK_TABLE, SEQ_LENS, OUT,
        stride_qb, stride_qh, stride_qd,
        stride_kp, stride_kh, stride_ks, stride_kd,
        stride_vp, stride_vh, stride_vs, stride_vd,
        stride_bt_b, stride_bt_n,
        stride_ob, stride_oh, stride_od,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_id = pid // NUM_HEADS
        head_id = pid % NUM_HEADS
        seq_len = tl.load(SEQ_LENS + batch_id)
        d = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q + batch_id * stride_qb + head_id * stride_qh + d * stride_qd,
            mask=d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)

        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        running_acc = tl.full((BLOCK_D,), 0.0, tl.float32)

        num_blocks = tl.cdiv(seq_len, PAGE_SIZE)
        for logical_block in range(MAX_BLOCKS):
            if logical_block >= num_blocks:
                break
            physical_page = tl.load(
                BLOCK_TABLE + batch_id * stride_bt_b + logical_block * stride_bt_n
            )
            if physical_page < 0:
                break
            for page_offset in tl.static_range(0, PAGE_SIZE):
                token_pos = logical_block * PAGE_SIZE + page_offset
                if token_pos < seq_len:
                    k = tl.load(
                        K
                        + physical_page * stride_kp
                        + head_id * stride_kh
                        + page_offset * stride_ks
                        + d * stride_kd,
                        mask=d < HEAD_DIM,
                        other=0.0,
                    ).to(tl.float32)
                    v = tl.load(
                        V
                        + physical_page * stride_vp
                        + head_id * stride_vh
                        + page_offset * stride_vs
                        + d * stride_vd,
                        mask=d < HEAD_DIM,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.sum(q * k, axis=0) * SCALE
                    new_max = tl.maximum(running_max, score)
                    old_scale = tl.exp(running_max - new_max)
                    new_scale = tl.exp(score - new_max)
                    running_acc = running_acc * old_scale + v * new_scale
                    running_sum = running_sum * old_scale + new_scale
                    running_max = new_max

        out = running_acc / tl.maximum(running_sum, 1e-6)
        tl.store(
            OUT + batch_id * stride_ob + head_id * stride_oh + d * stride_od,
            out,
            mask=d < HEAD_DIM,
        )


def _torch_paged_decode_flat(q, k_cache, v_cache, block_table, seq_lens, page_size, scale):
    b, h, tq, d = q.shape
    assert tq == 1
    outs = []
    for i in range(b):
        length = int(seq_lens[i].item())
        n_blocks = (length + page_size - 1) // page_size if length > 0 else 0
        ks, vs = [], []
        for bi in range(n_blocks):
            page = int(block_table[i, bi].item())
            if page < 0:
                break
            ks.append(k_cache[page])  # [H, PAGE, D]
            vs.append(v_cache[page])
        if not ks:
            outs.append(torch.zeros_like(q[i : i + 1]))
            continue
        k = torch.cat(ks, dim=1)[:, :length].unsqueeze(0)  # [1,H,L,D]
        v = torch.cat(vs, dim=1)[:, :length].unsqueeze(0)
        qi = q[i : i + 1]
        outs.append(
            F.scaled_dot_product_attention(qi, k, v, is_causal=False, scale=scale)
        )
    return torch.cat(outs, dim=0)


def forge_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Forge attention entry: Triton on CUDA, torch gather otherwise."""
    global LAST_FORGE_ATTN_BACKEND, LAST_FORGE_ATTN_ERROR
    LAST_FORGE_ATTN_ERROR = None
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("q must be [B,H,1,D]")
    b, h, _, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)

    use_triton = (
        TRITON_AVAILABLE
        and q.is_cuda
        and k_cache.is_cuda
        and d <= 128
    )
    if use_triton:
        try:
            out = torch.empty_like(q)
            q3 = q.contiguous().squeeze(2)
            out3 = out.squeeze(2)
            max_blocks = block_table.shape[1]
            _forge_paged_decode_kernel[(b * h,)](
                q3,
                k_cache,
                v_cache,
                block_table.contiguous(),
                seq_lens.contiguous(),
                out3,
                q3.stride(0),
                q3.stride(1),
                q3.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(3),
                block_table.stride(0),
                block_table.stride(1),
                out3.stride(0),
                out3.stride(1),
                out3.stride(2),
                h,
                d,
                page_size,
                max_blocks,
                scale,
                triton.next_power_of_2(d),
                num_warps=4,
            )
            LAST_FORGE_ATTN_BACKEND = "triton_paged_decode"
            return out
        except Exception as exc:
            LAST_FORGE_ATTN_ERROR = repr(exc)

    LAST_FORGE_ATTN_BACKEND = "torch_paged_decode"
    return _torch_paged_decode_flat(
        q, k_cache, v_cache, block_table, seq_lens, page_size, scale
    )


def get_last_forge_attn_backend():
    return LAST_FORGE_ATTN_BACKEND, LAST_FORGE_ATTN_ERROR
