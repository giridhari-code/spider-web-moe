"""Unit tests: HTTP InferenceServer (in-process, no bind)."""

from __future__ import annotations

import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine
from spider_infer.server import InferenceServer, ServeConfig, _UI_HTML, _sse


def _srv(max_inflight=4):
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
    engine = LLMEngine(model, eng)
    return InferenceServer(
        engine,
        ServeConfig(max_inflight=max_inflight, default_max_tokens=4, hard_max_tokens=16),
    )


def test_generate_text_shape():
    srv = _srv()
    out = srv.generate_text("hello", max_tokens=3, temperature=0.0)
    assert out["object"] == "text_completion"
    assert out["usage"]["completion_tokens"] == 3
    assert "choices" in out and len(out["choices"]) == 1
    assert out["latency_ms"] >= 0


def test_stream_tokens_count():
    srv = _srv()
    pieces, usage = [], None
    for piece, u in srv.stream_tokens("hi", max_tokens=4, temperature=0.0):
        if u is not None:
            usage = u
        elif piece:
            pieces.append(piece)
    assert usage["completion_tokens"] == 4
    assert len(pieces) == 4


def test_hard_max_tokens_clamped():
    srv = _srv()
    out = srv.generate_text("x", max_tokens=9999, temperature=0.0)
    assert out["usage"]["completion_tokens"] <= 16


def test_capacity_429():
    srv = _srv(max_inflight=1)
    assert srv._sem.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="capacity"):
            srv.generate_text("blocked", max_tokens=1)
        snap = srv.metrics.snapshot()
        assert snap["requests_rejected"] >= 1
    finally:
        srv._sem.release()


def test_metrics_after_success():
    srv = _srv()
    before = srv.metrics.snapshot()["requests_success"]
    srv.generate_text("abc", max_tokens=2, temperature=0.0)
    after = srv.metrics.snapshot()
    assert after["requests_success"] == before + 1
    assert after["tokens_generated"] >= 2
    prom = srv.metrics.prometheus()
    assert "spider_requests_success" in prom


def test_sse_helper_format():
    raw = _sse({"type": "RUN_STARTED"})
    assert raw.startswith(b"data: ")
    assert raw.endswith(b"\n\n")
    assert b"RUN_STARTED" in raw


def test_ui_serves_ag_ui_client():
    assert "SpiderInfer" in _UI_HTML
    assert "/agent" in _UI_HTML
    assert "TEXT_MESSAGE_CONTENT" in _UI_HTML
