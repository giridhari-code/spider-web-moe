"""Unit tests: SpiderInfer LLMEngine core behaviour."""

from __future__ import annotations

import torch
import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams
from spider_infer.sequence import Sequence, SequenceStatus


def _cfg(**kw):
    base = dict(
        vocab_size=64, hidden_size=32, num_heads=2, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64, use_rope=True,
    )
    base.update(kw)
    return ModelConfig(**base).validate()


def _engine(chunk_size=8, device="cpu"):
    model = SpiderWebLM(_cfg()).eval()
    eng = EngineConfig.default_for_model(max_seq_len=64, device=device)
    eng.cache.num_pages = 128
    eng.cache.dtype = "float32"
    eng.scheduler.prefill_chunk_size = chunk_size
    eng.scheduler.max_num_seqs = 4
    eng.enable_cuda_graph = False
    return LLMEngine(model, eng, enable_prefix_cache=True)


def test_generate_single_request_length():
    eng = _engine()
    outs = eng.generate([[1, 2, 3]], SamplingParams(max_tokens=5, temperature=0.0))
    assert len(outs) == 1
    assert len(outs[0].output_token_ids) == 5
    assert outs[0].finished
    assert outs[0].prompt_token_ids == [1, 2, 3]


def test_generate_batch_independent():
    eng = _engine()
    outs = eng.generate(
        [[1, 2], [10, 11, 12]],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    assert len(outs) == 2
    assert all(len(o.output_token_ids) == 2 for o in outs)


def test_greedy_outputs_valid_token_ids():
    eng = _engine()
    p = SamplingParams(max_tokens=4, temperature=0.0, seed=0)
    out = eng.generate([[5, 6, 7]], p)[0]
    assert len(out.output_token_ids) == 4
    vocab = eng.model.cfg.vocab_size
    assert all(0 <= t < vocab for t in out.output_token_ids)


def test_stream_matches_generate_length():
    eng = _engine()
    tokens = list(
        eng.generate_stream([1, 2, 3, 4], SamplingParams(max_tokens=3, temperature=0.0))
    )
    assert len(tokens) == 3


def test_chunked_prefill_long_prompt():
    eng = _engine(chunk_size=4)
    prompt = list(range(1, 20))
    outs = eng.generate([prompt], SamplingParams(max_tokens=2, temperature=0.0))
    assert outs[0].prompt_token_ids == prompt
    assert len(outs[0].output_token_ids) == 2


def test_stats_keys():
    eng = _engine()
    eng.generate([[1, 2]], SamplingParams(max_tokens=1, temperature=0.0))
    s = eng.stats()
    assert "cache" in s
    assert "waiting" in s
    assert "running" in s


def test_oversized_prompt_rejected():
    eng = _engine()
    with pytest.raises(ValueError):
        eng.scheduler.add_request(
            Sequence(
                prompt_token_ids=list(range(200)),
                sampling_params=SamplingParams(max_tokens=1),
            )
        )


def test_from_checkpoint_roundtrip(tmp_path):
    model = SpiderWebLM(_cfg()).eval()
    path = tmp_path / "m.pt"
    torch.save({"model_config": model.cfg.__dict__, "model": model.state_dict()}, path)
    eng_cfg = EngineConfig.default_for_model(max_seq_len=64, device="cpu")
    eng_cfg.cache.dtype = "float32"
    eng_cfg.cache.num_pages = 64
    engine = LLMEngine.from_checkpoint(path, engine_config=eng_cfg, device="cpu")
    outs = engine.generate([[1, 2, 3]], SamplingParams(max_tokens=2, temperature=0.0))
    assert len(outs[0].output_token_ids) == 2
