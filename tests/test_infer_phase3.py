"""Phase-3: chunked prefill + preemption tests."""

from __future__ import annotations

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams


def _engine(chunk_size=4, max_seq_len=64, num_pages=128):
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=2,
        num_experts=2,
        top_k=1,
        edges_per_expert=1,
        num_loops=1,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=max_seq_len,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    engine_cfg = EngineConfig.default_for_model(max_seq_len=max_seq_len, device="cpu")
    engine_cfg.cache.num_pages = num_pages
    engine_cfg.cache.dtype = "float32"
    engine_cfg.scheduler.max_num_seqs = 4
    engine_cfg.scheduler.prefill_chunk_size = chunk_size
    engine_cfg.scheduler.enable_preemption = True
    return LLMEngine(model, engine_cfg)


def test_chunked_prefill_long_prompt():
    engine = _engine(chunk_size=4)
    prompt = list(range(1, 20))  # 19 tokens → multiple chunks
    outs = engine.generate(
        [prompt],
        SamplingParams(max_tokens=3, temperature=0.0),
    )
    assert len(outs) == 1
    assert len(outs[0].output_token_ids) == 3
    assert outs[0].prompt_token_ids == prompt


def test_chunked_batch():
    engine = _engine(chunk_size=3)
    outs = engine.generate(
        [list(range(1, 10)), list(range(20, 28))],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    assert len(outs) == 2
    assert all(len(o.output_token_ids) == 2 for o in outs)


def test_stream_with_chunks():
    engine = _engine(chunk_size=2)
    tokens = list(
        engine.generate_stream(
            list(range(1, 9)),
            SamplingParams(max_tokens=3, temperature=0.0),
        )
    )
    assert len(tokens) == 3


def test_phase1_still_works_unlimited_chunk():
    engine = _engine(chunk_size=0)  # 0 → unlimited
    outs = engine.generate(
        [[1, 2, 3]],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    assert len(outs[0].output_token_ids) == 2
