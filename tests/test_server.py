"""In-process tests for InferenceServer (REST + streaming)."""

from __future__ import annotations

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine
from spider_infer.server import InferenceServer, ServeConfig, _UI_HTML


def _engine():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=2, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    eng = EngineConfig.default_for_model(max_seq_len=64, device="cpu")
    eng.cache.num_pages = 64
    eng.cache.dtype = "float32"
    eng.scheduler.prefill_chunk_size = 16
    return LLMEngine(model, eng)


def test_server_generate_text():
    engine = _engine()
    srv = InferenceServer(engine, ServeConfig(max_inflight=2, default_max_tokens=4))
    out = srv.generate_text("hi", max_tokens=3, temperature=0.0)
    assert out["object"] == "text_completion"
    assert out["usage"]["completion_tokens"] == 3
    assert out["choices"][0]["text"] is not None


def test_server_stream_tokens():
    engine = _engine()
    srv = InferenceServer(engine, ServeConfig(max_inflight=2))
    pieces = []
    usage = None
    for piece, u in srv.stream_tokens("hello", max_tokens=4, temperature=0.0):
        if u is not None:
            usage = u
        elif piece:
            pieces.append(piece)
    assert usage is not None
    assert usage["completion_tokens"] == 4
    assert len(pieces) == 4


def test_server_capacity_reject():
    engine = _engine()
    srv = InferenceServer(engine, ServeConfig(max_inflight=1))
    assert srv._sem.acquire(blocking=False)
    try:
        import pytest
        with pytest.raises(RuntimeError, match="capacity"):
            srv.generate_text("x", max_tokens=1)
    finally:
        srv._sem.release()


def test_metrics_prometheus():
    engine = _engine()
    srv = InferenceServer(engine, ServeConfig())
    srv.generate_text("abc", max_tokens=2, temperature=0.0)
    text = srv.metrics.prometheus()
    assert "spider_requests_success" in text
    assert "spider_tokens_generated" in text


def test_ui_html_present():
    assert "SpiderInfer" in _UI_HTML
    assert "/agent" in _UI_HTML
    assert "TEXT_MESSAGE_CONTENT" in _UI_HTML
