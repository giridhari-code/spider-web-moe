"""Forge 0.2 — Triton paged decode + expert GEMM (CPU fallback ok)."""
from __future__ import annotations

import math

import torch

from spider_infer.forge import FORGE_VERSION, ForgeConfig, ForgeBackend, init_forge
from spider_infer.forge.attention.triton_kernels import (
    forge_paged_decode,
    get_last_forge_attn_backend,
)
from spider_infer.forge.gemm.triton_kernels import (
    triton_linear,
    triton_expert_ffn,
    get_last_forge_gemm_backend,
)


def test_forge_version_0_2():
    assert tuple(map(int, FORGE_VERSION.split(".")[:2])) >= (0, 2)


def test_paged_decode_matches_manual_attention():
    torch.manual_seed(0)
    b, h, d, page, n_pages = 2, 2, 8, 4, 4
    # 2 tokens per seq → 1 page each
    q = torch.randn(b, h, 1, d)
    k_cache = torch.randn(n_pages, h, page, d)
    v_cache = torch.randn(n_pages, h, page, d)
    block_table = torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=torch.int32)
    seq_lens = torch.tensor([3, 2], dtype=torch.int32)

    out = forge_paged_decode(q, k_cache, v_cache, block_table, seq_lens, page_size=page)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    backend, _ = get_last_forge_attn_backend()
    assert backend in ("torch_paged_decode", "triton_paged_decode")

    # Manual check row 0
    scale = 1.0 / math.sqrt(d)
    k0 = k_cache[0, :, :3, :]  # H,L,D
    v0 = v_cache[0, :, :3, :]
    qi = q[0]  # H,1,D
    scores = torch.einsum("hid,hjd->hij", qi, k0) * scale
    weights = torch.softmax(scores, dim=-1)
    ref = torch.einsum("hij,hjd->hid", weights, v0).unsqueeze(0)
    assert torch.allclose(out[0:1], ref, rtol=1e-4, atol=1e-4)


def test_triton_linear_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(5, 16)
    w = torch.randn(32, 16)
    b = torch.randn(32)
    y = triton_linear(x, w, b)
    ref = torch.nn.functional.linear(x, w, b)
    assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)
    backend, _ = get_last_forge_gemm_backend()
    assert backend in ("torch_linear", "triton_gemm")


def test_expert_ffn_finite():
    x = torch.randn(7, 16)
    w_up = torch.randn(64, 16)
    w_down = torch.randn(16, 64)
    y = triton_expert_ffn(x, w_up, None, w_down, None)
    assert y.shape == (7, 16)
    assert torch.isfinite(y).all()


def test_forge_runtime_uses_0_2_impls():
    rt = init_forge(ForgeConfig(device="cpu", backend=ForgeBackend.TORCH))
        impl = rt.attention.stats()["impl"]
    assert any(s in impl for s in ("triton", "torch", "paged", "mla"))
    assert rt.gemm.stats()["forge_version_target"] >= "0.2"
    x = torch.randn(3, 8)
    w = torch.randn(16, 8)
    assert rt.gemm.linear(x, w).shape == (3, 16)
