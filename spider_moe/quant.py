"""Quantization helpers: scaled FP8 KV and group-wise INT4 weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Prefer native FP8 when available (PyTorch 2.1+ / 2.4+ CUDA).
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# Max representable finite magnitude for E4M3 (approx).
FP8_E4M3_MAX = 448.0


def fp8_available() -> bool:
    return _FP8_DTYPE is not None


def fp8_dtype() -> torch.dtype:
    if _FP8_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
    return _FP8_DTYPE


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return _FP8_DTYPE is not None and dtype == _FP8_DTYPE


def compute_fp8_scale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-tensor scale: max(|x|) / FP8_MAX so quantized values stay in range."""
    amax = x.detach().float().abs().amax()
    return (amax / FP8_E4M3_MAX).clamp_min(eps)


def quantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-aware FP8 quantize.

    Returns (fp8_tensor, scale) where scale is a 0-dim float32 tensor on x.device.
    If scale is None it is computed from x (per-tensor amax).
    """
    if scale is None:
        scale_t = compute_fp8_scale(x)
    elif isinstance(scale, torch.Tensor):
        scale_t = scale.detach().float().reshape(()).to(device=x.device)
    else:
        scale_t = torch.tensor(float(scale), dtype=torch.float32, device=x.device)
    # Avoid div-by-zero; scale already clamped in compute_fp8_scale.
    y = (x.float() / scale_t).to(fp8_dtype())
    return y, scale_t


def dequantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor | float | None = None,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """FP8 → compute dtype. If scale is None, behaves like a plain cast (scale=1)."""
    y = x.to(torch.float32)
    if scale is None:
        return y.to(compute_dtype)
    if isinstance(scale, torch.Tensor):
        scale_t = scale.to(device=y.device, dtype=torch.float32)
    else:
        scale_t = torch.tensor(float(scale), dtype=torch.float32, device=y.device)
    return (y * scale_t).to(compute_dtype)


# ---------------------------------------------------------------------------
# Group-wise symmetric INT4 (packed 2 values per uint8)
# ---------------------------------------------------------------------------

@dataclass
class Int4Tensor:
    """Packed INT4 weights with per-group scales."""

    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_int4(self.packed, self.scales, self.shape, self.group_size, dtype=dtype)


def quantize_int4(
    weight: torch.Tensor,
    group_size: int = 64,
) -> Int4Tensor:
    """Symmetric per-group INT4 quantization along the last dimension."""
    if weight.ndim < 1:
        raise ValueError("weight must be at least 1D")
    *leading, n = weight.shape
    if n % 2 != 0:
        raise ValueError("last dimension must be even for INT4 packing")
    if group_size <= 0 or n % group_size != 0:
        group_size = n

    w = weight.detach().float().reshape(-1, n)
    g = group_size
    w_g = w.view(w.shape[0], n // g, g)
    scales = w_g.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    q = torch.round(w_g / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(w.shape[0], n)
    even = (q[:, 0::2] & 0x0F).to(torch.uint8)
    odd = (q[:, 1::2] & 0x0F).to(torch.uint8)
    packed = (even | (odd << 4)).view(*leading, n // 2)
    scales = scales.view(*leading, n // g)
    return Int4Tensor(
        packed=packed.contiguous(),
        scales=scales.contiguous(),
        shape=tuple(weight.shape),
        group_size=g,
    )


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
    group_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    *leading, half = packed.shape
    n = half * 2
    flat = packed.reshape(-1, half)
    even = (flat & 0x0F).to(torch.int8)
    odd = ((flat >> 4) & 0x0F).to(torch.int8)
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    q = torch.empty(flat.shape[0], n, device=packed.device, dtype=torch.int8)
    q[:, 0::2] = even
    q[:, 1::2] = odd
    g = group_size
    q_g = q.view(q.shape[0], n // g, g).float()
    sc = scales.reshape(q.shape[0], n // g).unsqueeze(-1)
    w = (q_g * sc).view(*leading, n)
    if dtype is not None:
        w = w.to(dtype)
    return w.view(shape)


def quantize_fp8_with_scale_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize x using a broadcastable scale tensor (not necessarily scalar).

    x: [..., D] or [H, D]
    scale: broadcastable to x (e.g. [H, 1] for per-head)
    """
    scale_b = scale.to(device=x.device, dtype=torch.float32)
    while scale_b.ndim < x.ndim:
        scale_b = scale_b.unsqueeze(-1)
    return (x.float() / scale_b.clamp_min(1e-12)).to(fp8_dtype())


def dequantize_fp8_with_scale_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    scale_b = scale.to(device=x.device, dtype=torch.float32)
    while scale_b.ndim < x.ndim:
        scale_b = scale_b.unsqueeze(-1)
    return (x.float() * scale_b).to(compute_dtype)
