"""Phase-1 SpiderInfer engine tests."""

from __future__ import annotations

import torch
import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams
from spider_infer.memory.block_manager import AllocStatus


def _tiny_engine(device="cpu"):
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
        max_seq_len=32,
    ).validate()
    model = SpiderWebLM(cfg).to(device).eval()
    engine_cfg = EngineConfig.default_for_model(max_seq_len=32, device=device)
    engine_cfg.cache.num_pages = 64
    engine_cfg.cache.dtype = "float32"
    engine_cfg.scheduler.max_num_seqs = 4
    return LLMEngine(model, engine_cfg)


def test_block_manager_can_allocate():
    engine = _tiny_engine()
    bm = engine.block_manager
    assert bm.can_allocate(8) is AllocStatus.OK
    assert bm.free_slots() > 0
    assert 0.0 <= bm.usage_ratio() <= 1.0


def test_generate_single():
    engine = _tiny_engine()
    outs = engine.generate(
        [[1, 2, 3]],
        SamplingParams(max_tokens=4, temperature=0.0, seed=0),
    )
    assert len(outs) == 1
    assert len(outs[0].output_token_ids) == 4
    assert outs[0].finished


def test_generate_batch():
    engine = _tiny_engine()
    outs = engine.generate(
        [[1, 2, 3], [4, 5]],
        SamplingParams(max_tokens=3, temperature=0.0),
    )
    assert len(outs) == 2
    assert all(len(o.output_token_ids) == 3 for o in outs)


def test_stream():
    engine = _tiny_engine()
    tokens = list(
        engine.generate_stream(
            [7, 8, 9],
            SamplingParams(max_tokens=3, temperature=0.0),
        )
    )
    assert len(tokens) == 3


def test_stop_token():
    engine = _tiny_engine()
    # Force a stop by setting max_tokens small; also verify stop_token_ids path exists.
    outs = engine.generate(
        [[1, 2]],
        SamplingParams(max_tokens=2, stop_token_ids=[99999], temperature=0.0),
    )
    assert len(outs[0].output_token_ids) <= 2
