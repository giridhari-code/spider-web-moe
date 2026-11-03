"""Full pipeline integration tests."""

from __future__ import annotations

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams
from spider_infer.checkpoint import load_checkpoint


def _tiny(device="cpu"):
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=2, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64,
    ).validate()
    model = SpiderWebLM(cfg).to(device).eval()
    eng = EngineConfig.default_for_model(max_seq_len=64, device=device)
    eng.cache.num_pages = 128
    eng.cache.dtype = "float32"
    eng.scheduler.prefill_chunk_size = 8
    eng.scheduler.max_num_seqs = 4
    return model, eng


def test_chunked_and_batch():
    model, eng = _tiny()
    engine = LLMEngine(model, eng)
    outs = engine.generate(
        [list(range(1, 25)), list(range(2, 10))],
        SamplingParams(max_tokens=3, temperature=0.0),
    )
    assert len(outs) == 2
    assert all(len(o.output_token_ids) == 3 for o in outs)


def test_prefix_cache_concurrent_duplicate():
    model, eng = _tiny()
    engine = LLMEngine(model, eng, enable_prefix_cache=True)
    prompt = list(range(1, 16))
    # Same prompt twice in one batch — second may hit after first prefill lands.
    outs = engine.generate(
        [prompt, prompt],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    assert len(outs) == 2
    assert len(outs[0].output_token_ids) == 2


def test_checkpoint_roundtrip(tmp_path):
    model, eng = _tiny()
    ckpt_path = tmp_path / "m.pt"
    torch.save({"model_config": model.cfg.__dict__, "model": model.state_dict()}, ckpt_path)
    loaded, cfg, _ = load_checkpoint(ckpt_path, device="cpu")
    engine = LLMEngine(loaded, eng)
    outs = engine.generate([[1, 2, 3]], SamplingParams(max_tokens=2, temperature=0.0))
    assert len(outs[0].output_token_ids) == 2


def test_from_checkpoint(tmp_path):
    model, eng = _tiny()
    ckpt_path = tmp_path / "m.pt"
    torch.save({"model_config": model.cfg.__dict__, "model": model.state_dict()}, ckpt_path)
    engine = LLMEngine.from_checkpoint(ckpt_path, engine_config=eng, device="cpu")
    outs = engine.generate([[4, 5, 6]], SamplingParams(max_tokens=2, temperature=0.0))
    assert len(outs[0].output_token_ids) == 2
