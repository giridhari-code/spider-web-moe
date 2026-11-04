"""Forge 0.3 — FP8 GEMM path (DeepGEMM-shaped).

Weights (and optionally activations) are quantised to float8_e4m3fn with
per-tensor scales, matmul runs in FP16/BF16 after dequant (portable).
When native FP8 matmul lands on the device, swap only this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from spider_moe.quant import (
    dequantize_fp8,
    fp8_available,
    quantize_fp8,
)

LAST_FP8_GEMM_BACKEND = "not_run"
LAST_FP8_GEMM_ERROR: str | None = None


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    quantize_input: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Linear with FP8-quantised weights (and optional input).

    Returns same shape as F.linear(x, weight, bias). Numerics ≈ FP16 linear
    within quantisation noise.
    """
    global LAST_FP8_GEMM_BACKEND, LAST_FP8_GEMM_ERROR
    LAST_FP8_GEMM_ERROR = None

    if not fp8_available():
        LAST_FP8_GEMM_BACKEND = "torch_linear_no_fp8"
        return F.linear(x, weight, bias)

    try:
        compute_dtype = compute_dtype or (
            torch.float16 if x.dtype in (torch.float16, torch.bfloat16) else torch.float32
        )
        w_fp8, w_scale = quantize_fp8(weight)
        w_deq = dequantize_fp8(w_fp8, w_scale, compute_dtype=compute_dtype)

        if quantize_input:
            x_fp8, x_scale = quantize_fp8(x)
            x_deq = dequantize_fp8(x_fp8, x_scale, compute_dtype=compute_dtype)
        else:
            x_deq = x.to(dtype=compute_dtype)

        out = F.linear(x_deq, w_deq, None if bias is None else bias.to(dtype=compute_dtype))
        out = out.to(dtype=x.dtype)
        LAST_FP8_GEMM_BACKEND = "fp8_weight_linear"
        return out
    except Exception as exc:
        LAST_FP8_GEMM_ERROR = repr(exc)
        LAST_FP8_GEMM_BACKEND = "torch_linear_fp8_fallback"
        return F.linear(x, weight, bias)


def fp8_expert_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    *,
    quantize_input: bool = False,
) -> torch.Tensor:
    h = fp8_linear(x, w_up, b_up, quantize_input=quantize_input)
    h = F.gelu(h)
    return fp8_linear(h, w_down, b_down, quantize_input=False)


def get_last_fp8_gemm_backend():
    return LAST_FP8_GEMM_BACKEND, LAST_FP8_GEMM_ERROR
