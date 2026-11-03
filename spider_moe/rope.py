"""Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import torch


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return cos/sin table of shape [max_seq_len, head_dim]."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=dtype) / half)
    )
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [S, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position: int = 0,
) -> torch.Tensor:
    """Apply RoPE to tensor x of shape [B, H, T, D].

    cos/sin: [S, D] full table; uses slice [position, position+T).
    """
    b, h, t, d = x.shape
    cos_t = cos[position : position + t].to(dtype=x.dtype, device=x.device)
    sin_t = sin[position : position + t].to(dtype=x.dtype, device=x.device)
    # [T, D] -> [1, 1, T, D]
    cos_t = cos_t.view(1, 1, t, d)
    sin_t = sin_t.view(1, 1, t, d)
    return x * cos_t + _rotate_half(x) * sin_t


def apply_rope_positions(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE with per-row absolute positions.

    x: [B, H, T, D]
    positions: [B, T] int64
    """
    b, h, t, d = x.shape
    # Gather cos/sin per token: [B, T, D]
    cos_t = cos[positions]  # uses advanced indexing
    sin_t = sin[positions]
    cos_t = cos_t.to(dtype=x.dtype, device=x.device).view(b, 1, t, d)
    sin_t = sin_t.to(dtype=x.dtype, device=x.device).view(b, 1, t, d)
    return x * cos_t + _rotate_half(x) * sin_t


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)
