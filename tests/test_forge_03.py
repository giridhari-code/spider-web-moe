"""Forge 0.3 — MLA latent attention + FP8 GEMM."""
from __future__ import annotations

import torch

from spider_infer.forge import FORGE_VERSION, ForgeConfig, ForgeBackend, init_forge
from spider_infer.forge.attention.mla import mla_decode, get_last_mla_backend
from spider_infer.forge.gemm.fp8 import fp8_linear, fp8_expert_ffn, get_last_fp8_gemm_backend
from spider_moe.quant import fp8_available


def test_forge_version_0_3():
    assert tuple(map(int, FORGE_VERSION.split(".")[:2])) >= (0, 3)


def test_mla_decode_shape_and_finite():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)
    out = mla_decode(q, k, v, latent_rank=8)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    backend, _ = get_last_mla_backend()
    assert backend in ("mla_latent_torch", "mla_fallback_sdpa")


def test_mla_via_forge_attention():
    rt = init_forge(ForgeConfig(device="cpu", attention_use_mla=True, mla_latent_rank=8))
    q = torch.randn(1, 2, 3, 16)
    k = torch.randn(1, 2, 5, 16)
    v = torch.randn(1, 2, 5, 16)
    out = rt.attention.decode(q, k, v, causal=True)
    assert out.shape == q.shape
    assert rt.attention.stats()["forge_version_target"] == "0.3"
    out2 = rt.attention.decode_mla(q, k, v, latent_rank=4)
    assert out2.shape == q.shape


def test_fp8_linear_close_to_fp32():
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    w = torch.randn(32, 16)
    b = torch.randn(32)
    y = fp8_linear(x, w, b)
    ref = torch.nn.functional.linear(x, w, b)
    assert y.shape == ref.shape
    # quantisation noise — not exact
    if fp8_available():
        assert (y - ref).abs().mean() < 0.5
        backend, _ = get_last_fp8_gemm_backend()
        assert "fp8" in backend or "fallback" in backend or "no_fp8" in backend
    else:
        assert torch.allclose(y, ref)


def test_fp8_expert_ffn_and_config_flag():
    rt = init_forge(ForgeConfig(device="cpu", gemm_fp8=True, backend=ForgeBackend.TORCH))
    x = torch.randn(5, 16)
    w_up = torch.randn(64, 16)
    w_down = torch.randn(16, 64)
    y = rt.gemm.expert_ffn(x, w_up, None, w_down, None)
    assert y.shape == (5, 16)
    assert torch.isfinite(y).all()
    assert rt.gemm.stats()["fp8"] is True
    y2 = rt.gemm.linear_fp8(x, w_up)
    assert y2.shape == (5, 64)
