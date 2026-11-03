
"""FP8 scale granularity modes: page / head / token / token_head."""

from __future__ import annotations

import torch
import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.quant import fp8_available, fp8_dtype


def _tiny():
    return ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, use_rope=True,
    ).validate()


@pytest.mark.skipif(not fp8_available(), reason="FP8 not available")
@pytest.mark.parametrize("mode,scale_ndim", [
    ("page", 2),
    ("head", 3),
    ("token", 3),
    ("token_head", 4),
])
def test_fp8_scale_mode_shapes_and_generate(mode, scale_ndim):
    model = SpiderWebLM(_tiny()).eval()
    cache = model.allocate_global_paged_cache(
        num_pages=16, page_size=4, max_requests=2,
        dtype=fp8_dtype(), device="cpu", fp8_scale_mode=mode,
    )
    assert cache.fp8_scale_mode == mode
    assert cache.key_scales.ndim == scale_ndim
    x = torch.randint(0, 64, (1, 7))
    logits, cache = model.prefill_paged(x, cache, request_ids=["r0"])
    assert torch.isfinite(logits).all()
    assert cache.sequence_length("r0") == 7
    y = torch.randint(0, 64, (1, 1))
    logits2, cache = model.decode_step_paged(y, cache, ["r0"])
    assert torch.isfinite(logits2).all()
    assert (cache.key_scales > 0).all()


@pytest.mark.skipif(not fp8_available(), reason="FP8 not available")
def test_engine_token_head_mode():
    from spider_infer import EngineConfig, LLMEngine, SamplingParams
    model = SpiderWebLM(_tiny()).eval()
    eng = EngineConfig.default_for_model(max_seq_len=32, device="cpu")
    eng.cache.dtype = "fp8"
    eng.cache.fp8_scale_mode = "token_head"
    eng.cache.num_pages = 32
    eng.scheduler.prefill_chunk_size = 4
    engine = LLMEngine(model, eng)
    outs = engine.generate([[1, 2, 3, 4, 5]], SamplingParams(max_tokens=2, temperature=0.0))
    assert len(outs[0].output_token_ids) == 2
    assert engine.block_manager.cache.fp8_scale_mode == "token_head"
