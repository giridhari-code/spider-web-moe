"""Unit tests: AG-UI 1.0 protocol helpers + end-to-end event stream."""

from __future__ import annotations

import json

import pytest

from spider_infer import ag_ui
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine
from spider_infer.server import InferenceServer, ServeConfig


def test_parse_run_input_defaults():
    tid, rid, msgs, opts = ag_ui.parse_run_input(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert tid.startswith("thread-") or tid
    assert rid.startswith("run-") or rid
    assert len(msgs) == 1
    assert "temperature" in opts


def test_parse_run_input_explicit():
    tid, rid, msgs, opts = ag_ui.parse_run_input(
        {
            "threadId": "t-9",
            "runId": "r-9",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 12,
            "temperature": 0.2,
        }
    )
    assert tid == "t-9"
    assert rid == "r-9"
    assert opts["max_tokens"] == 12
    assert opts["temperature"] == 0.2


def test_parse_run_input_bad_messages():
    with pytest.raises(ValueError):
        ag_ui.parse_run_input({"messages": "not-a-list"})


def test_messages_to_prompt_multiturn():
    p = ag_ui.messages_to_prompt(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"},
        ]
    )
    assert "system: be brief" in p
    assert "user: more" in p
    assert p.strip().endswith("assistant:")


def test_messages_content_parts():
    p = ag_ui.messages_to_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}]}]
    )
    assert "user: abcd" in p


def test_lifecycle_and_text_event_shapes():
    e = ag_ui.run_started("t", "r")
    assert e["type"] == "RUN_STARTED"
    assert set(["type", "timestamp", "threadId", "runId"]).issubset(e.keys())

    m = ag_ui.text_message_start("m1")
    assert m["role"] == "assistant"
    c = ag_ui.text_message_content("m1", "hi")
    assert c["delta"] == "hi"
    end = ag_ui.text_message_end("m1")
    assert end["type"] == "TEXT_MESSAGE_END"
    fin = ag_ui.run_finished("t", "r")
    assert fin["outcome"]["type"] == "success"
    err = ag_ui.run_error("boom", "t", "r")
    assert err["type"] == "RUN_ERROR"
    assert err["message"] == "boom"


def test_stream_text_run_ordering():
    def gen():
        yield "A", None
        yield "B", None
        yield "", {"completion_tokens": 2}

    types = [e["type"] for e in ag_ui.stream_text_run(gen(), thread_id="t", run_id="r")]
    assert types == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]


def test_stream_text_run_error_path():
    def gen():
        yield "x", None
        raise RuntimeError("fail-mid-stream")

    events = list(ag_ui.stream_text_run(gen(), thread_id="t", run_id="r"))
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_ERROR"
    assert "fail-mid-stream" in events[-1]["message"]


def test_e2e_engine_to_ag_ui_events():
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

    body = {
        "threadId": "thr-test",
        "runId": "run-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "temperature": 0.0,
    }
    tid, rid, msgs, opts = ag_ui.parse_run_input(body)
    prompt = ag_ui.messages_to_prompt(msgs)
    stream = srv.stream_tokens(
        prompt, max_tokens=opts["max_tokens"], temperature=0.0
    )
    events = list(ag_ui.stream_text_run(stream, thread_id=tid, run_id=rid, input_=body))

    # Wire format: each event JSON-serializable
    for e in events:
        json.dumps(e)

    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert events[0]["threadId"] == "thr-test"
    assert sum(1 for t in types if t == "TEXT_MESSAGE_CONTENT") == 3
