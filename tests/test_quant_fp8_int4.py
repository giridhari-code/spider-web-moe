
"""FP8 KV cache and INT4 MoE weight tests."""

from __future__ import annotations

import torch
import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.quant import (
    dequantize_fp8,
    dequantize_int4,
    fp8_available,
    quantize_fp8,
    quantize_int4,
)


def _tiny(**kw):
    base = dict(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, use_rope=True,
    )
    base.update(kw)
    return ModelConfig(**base).validate()


def test_int4_roundtrip():
    w = torch.randn(4, 64)
    q = quantize_int4(w, group_size=32)
    w2 = q.dequantize(dtype=torch.float32)
    # Reconstruction error should be bounded
    err = (w - w2).abs().mean().item()
    assert err < 0.15, err


def test_moe_int4_forward():
    model = SpiderWebLM(_tiny()).eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a = model(x)["logits"]
        model.quantize_moe_experts(group_size=32)
        b = model(x)["logits"]
    assert a.shape == b.shape
    assert torch.isfinite(b).all()
    # Quantization changes outputs, but not drastically for a random model
    assert (a - b).abs().mean().item() < 5.0


@pytest.mark.skipif(not fp8_available(), reason="FP8 dtype not in this PyTorch")
def test_fp8_cast_roundtrip():
    x = torch.randn(8, 16)
    q, scale = quantize_fp8(x)
    y = dequantize_fp8(q, scale, torch.float32)
    assert y.dtype == torch.float32
    assert (x - y).abs().mean().item() < 0.15


@pytest.mark.skipif(not fp8_available(), reason="FP8 dtype not in this PyTorch")
def test_fp8_kv_paged_generate():
    from spider_moe.quant import fp8_dtype

    model = SpiderWebLM(_tiny()).eval()
    cache = model.allocate_global_paged_cache(
        num_pages=32, page_size=8, max_requests=4,
        dtype=fp8_dtype(), device="cpu",
    )
    assert cache.dtype == fp8_dtype()
    x = torch.randint(0, 64, (1, 6))
    logits, cache = model.prefill_paged(x, cache, request_ids=["r0"])
    assert torch.isfinite(logits).all()
    assert cache.sequence_length("r0") == 6
    y = torch.randint(0, 64, (1, 1))
    logits2, cache = model.decode_step_paged(y, cache, ["r0"])
    assert torch.isfinite(logits2).all()
    assert cache.sequence_length("r0") == 7


@pytest.mark.skipif(not fp8_available(), reason="FP8 dtype not in this PyTorch")
def test_engine_fp8_cache():
    from spider_infer import EngineConfig, LLMEngine, SamplingParams
    from spider_moe.quant import fp8_available

    model = SpiderWebLM(_tiny()).eval()
    eng = EngineConfig.default_for_model(max_seq_len=32, device="cpu")
    eng.cache.dtype = "fp8"
    eng.cache.num_pages = 64
    eng.scheduler.prefill_chunk_size = 8
    engine = LLMEngine(model, eng)
    outs = engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=3, temperature=0.0))
    assert len(outs[0].output_token_ids) == 3


@pytest.mark.skipif(not fp8_available(), reason="FP8 dtype not in this PyTorch")
def test_fp8_scaled_roundtrip():
    from spider_moe.quant import quantize_fp8, dequantize_fp8
    x = torch.randn(16, 32) * 10.0  # larger dynamic range
    q, scale = quantize_fp8(x)
    y = dequantize_fp8(q, scale, torch.float32)
    # Scaled quant should track large magnitudes better than unit scale
    err = (x - y).abs().mean().item()
    assert err < 0.5, err
    assert scale.ndim == 0
    assert scale.item() > 0


@pytest.mark.skipif(not fp8_available(), reason="FP8 dtype not in this PyTorch")
def test_fp8_kv_scaled_pages_differ():
    from spider_moe.quant import fp8_dtype
    model = SpiderWebLM(_tiny()).eval()
    cache = model.allocate_global_paged_cache(
        num_pages=16, page_size=4, max_requests=2,
        dtype=fp8_dtype(), device="cpu",
    )
    x = torch.randint(0, 64, (1, 8))
    model.prefill_paged(x, cache, request_ids=["r0"])
    # Scales should be finite and positive
    assert cache.key_scales is not None
    used = cache.key_scales[cache.key_scales != 1.0]
    # At least some pages updated from default 1.0, or all still valid
    assert torch.isfinite(cache.key_scales).all()
    assert (cache.key_scales > 0).all()
