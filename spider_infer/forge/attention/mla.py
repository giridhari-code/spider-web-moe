"""Forge 0.3 — FlashMLA-style latent attention path.

Multi-Latent Attention idea (simplified, architecture-stable API):
  1. Compress K/V along head dim into a latent rank R << D (shared latent).
  2. Attend in latent space (cheaper KV bandwidth).
  3. Optionally up-project values back / keep V full-rank for quality.

This is not a bit-exact DeepSeek MLA clone; it freezes the *call contract*
Orbit/Forge use so a CUDA MLA kernel can replace the body later.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LAST_MLA_BACKEND = "not_run"
LAST_MLA_ERROR: str | None = None


def mla_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    latent_rank: int | None = None,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """MLA-style decode/attention.

    q,k,v: [B, H, T, D]  (Tq may be 1 for decode)
    Returns: [B, H, Tq, D]
    """
    global LAST_MLA_BACKEND, LAST_MLA_ERROR
    LAST_MLA_ERROR = None

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q,k,v must be [B,H,T,D]")
    b, h, tq, d = q.shape
    tk = k.shape[2]
    rank = latent_rank or max(1, d // 4)
    rank = min(rank, d)

    try:
        # Shared compression matrices (deterministic from dim — no extra params
        # for the reference path; production MLA uses learned / fused weights).
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x4D4C41 ^ d ^ rank)  # 'MLA'
        # Build on CPU then move for determinism across devices
        w_k = torch.randn(d, rank, generator=gen)
        w_v = torch.randn(d, rank, generator=gen)
        w_k = w_k.to(device=q.device, dtype=q.dtype)
        w_v = w_v.to(device=q.device, dtype=q.dtype)
        # Orthonormal-ish via QR for stability
        w_k, _ = torch.linalg.qr(w_k.float(), mode="reduced")
        w_v, _ = torch.linalg.qr(w_v.float(), mode="reduced")
        w_k = w_k.to(dtype=q.dtype)[:, :rank]
        w_v = w_v.to(dtype=q.dtype)[:, :rank]

        # Compress K/V: [B,H,T,D] @ [D,R] → [B,H,T,R]
        k_lat = torch.matmul(k, w_k)
        v_lat = torch.matmul(v, w_v)
        q_lat = torch.matmul(q, w_k)  # same basis as K for scores

        scale = scale if scale is not None else (1.0 / math.sqrt(rank))
        out_lat = F.scaled_dot_product_attention(
            q_lat, k_lat, v_lat, is_causal=causal, scale=scale
        )
        # Up-project latent V back to D
        out = torch.matmul(out_lat, w_v.transpose(0, 1))
        LAST_MLA_BACKEND = "mla_latent_torch"
        return out
    except Exception as exc:
        LAST_MLA_ERROR = repr(exc)
        LAST_MLA_BACKEND = "mla_fallback_sdpa"
        scale = scale if scale is not None else (1.0 / math.sqrt(d))
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def get_last_mla_backend():
    return LAST_MLA_BACKEND, LAST_MLA_ERROR
