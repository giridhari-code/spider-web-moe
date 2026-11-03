"""AG-UI 1.0 protocol producer tests."""

from __future__ import annotations

from spider_infer import ag_ui
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine
from spider_infer.server import InferenceServer, ServeConfig


def test_event_envelope():
    e = ag_ui.run_started("t1", "r1")
    assert e["type"] == "RUN_STARTED"
    assert e["threadId"] == "t1"
    assert e["runId"] == "r1"
    assert "timestamp" in e


def test_messages_to_prompt():
    p = ag_ui.messages_to_prompt(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert "user: hi" in p
    assert p.strip().endswith("assistant:")


def test_stream_text_run_order():
    def fake():
        yield "Hel", None
        yield "lo", None
        yield "", {"completion_tokens": 2}

    events = list(
        ag_ui.stream_text_run(fake(), thread_id="t", run_id="r", input_={"x": 1})
    )
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[1] == "TEXT_MESSAGE_START"
    assert types[2] == "TEXT_MESSAGE_CONTENT"
    assert types[3] == "TEXT_MESSAGE_CONTENT"
    assert types[4] == "TEXT_MESSAGE_END"
    assert types[5] == "RUN_FINISHED"
    assert events[2]["delta"] == "Hel"
    assert events[3]["delta"] == "lo"


def test_server_ag_ui_stream():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=2, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    eng = EngineConfig.default_for_model(max_seq_len=64, device="cpu")
    eng.cache.num_pages = 32
    eng.cache.dtype = "float32"
    engine = LLMEngine(model, eng)
    srv = InferenceServer(engine, ServeConfig(default_max_tokens=3))

    prompt = ag_ui.messages_to_prompt([{"role": "user", "content": "hi"}])
    token_stream = srv.stream_tokens(prompt, max_tokens=3, temperature=0.0)
    events = list(
        ag_ui.stream_text_run(token_stream, thread_id="thr", run_id="run1")
    )
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert "TEXT_MESSAGE_CONTENT" in types
    assert types[-1] == "RUN_FINISHED"
