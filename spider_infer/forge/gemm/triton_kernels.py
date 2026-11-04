"""Forge 0.2 — Triton expert GEMM helpers (DeepGEMM-shaped).

Provides a fused expert FFN path when CUDA+Triton are available; otherwise
torch.nn.functional.linear. The public contract matches ForgeGEMM.expert_ffn.
"""
from __future__ import annotations

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

LAST_FORGE_GEMM_BACKEND = "not_run"
LAST_FORGE_GEMM_ERROR: str | None = None


if TRITON_AVAILABLE:

    @triton.jit
    def _gemm_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_remaining = K - k0
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [M,K] @ weight.T [K,N] → [M,N] via Triton GEMM when on CUDA."""
    global LAST_FORGE_GEMM_BACKEND, LAST_FORGE_GEMM_ERROR
    LAST_FORGE_GEMM_ERROR = None
    if x.ndim != 2:
        x_flat = x.reshape(-1, x.shape[-1])
        out = triton_linear(x_flat, weight, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    m, k = x.shape
    n, k_w = weight.shape
    if k != k_w:
        raise ValueError("incompatible shapes for linear")

    use_triton = TRITON_AVAILABLE and x.is_cuda and weight.is_cuda and m > 0
    if use_triton:
        try:
            # weight is [N,K]; we need B as [K,N]
            w_t = weight.t().contiguous()
            out = torch.empty(m, n, device=x.device, dtype=torch.float32)
            block_m, block_n, block_k = 32, 32, 32
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
            _gemm_kernel[grid](
                x.contiguous().to(torch.float32),
                w_t.to(torch.float32),
                out,
                m, n, k,
                x.stride(0), x.stride(1),
                w_t.stride(0), w_t.stride(1),
                out.stride(0), out.stride(1),
                block_m, block_n, block_k,
            )
            out = out.to(dtype=x.dtype)
            if bias is not None:
                out = out + bias
            LAST_FORGE_GEMM_BACKEND = "triton_gemm"
            return out
        except Exception as exc:
            LAST_FORGE_GEMM_ERROR = repr(exc)

    LAST_FORGE_GEMM_BACKEND = "torch_linear"
    return F.linear(x, weight, bias)


def triton_expert_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
) -> torch.Tensor:
    h = triton_linear(x, w_up, b_up)
    h = F.gelu(h)
    return triton_linear(h, w_down, b_down)


def get_last_forge_gemm_backend():
    return LAST_FORGE_GEMM_BACKEND, LAST_FORGE_GEMM_ERROR
