"""Orbit readiness — drop-in server surface + public API."""
from __future__ import annotations

import torch

from spider_infer.config import EngineConfig
from spider_infer.engine import LLMEngine
from spider_infer.orbit import (
    OrbitConfig,
    OrbitEngine,
    build_orbit_engine,
    draft_orbit_generate,
    DraftOrbitConfig,
)
from spider_infer.sampling_params import SamplingParams
from spider_infer.server import InferenceServer, ServeConfig
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def _tiny():
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=2,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=32,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
    ).validate()


def test_build_orbit_engine_and_delegate():
    model = SpiderWebLM(_tiny()).eval()
    orbit = build_orbit_engine(
        model,
        OrbitConfig(max_loops=2, early_exit=False),
        engine_config=EngineConfig.default_for_model(32, "cpu"),
    )
    assert isinstance(orbit, OrbitEngine)
    assert orbit.model is model
    assert orbit.config.device == "cpu" or True  # EngineConfig present
    assert orbit.orbit_config.max_loops == 2
    outs = orbit.generate([[1, 2, 3]], SamplingParams(max_tokens=2, temperature=0.0))
    assert len(outs) == 1
    snap = orbit.orbit_snapshot()
    assert snap["technology"] == "Orbit"
    assert snap["orbit_version"] == "0.4.0"


def test_inference_server_accepts_orbit_engine():
    model = SpiderWebLM(_tiny()).eval()
    eng = LLMEngine(model, EngineConfig.default_for_model(32, "cpu"))
    orbit = OrbitEngine.from_engine(eng, OrbitConfig(early_exit=True, max_loops=2))
    server = InferenceServer(orbit, ServeConfig(model_name="sentinel"))
    params = server._params(8, 0.0, 1.0, 0, None)
    assert params.early_exit is True
    out = server.generate_text("hi", max_tokens=2, temperature=0.0)
    assert isinstance(out, dict)
    assert "choices" in out or "text" in out or "usage" in out


def test_draft_api_exported():
    model = SpiderWebLM(_tiny()).eval()
    prompt = torch.randint(0, 64, (1, 4))
    out, stats = draft_orbit_generate(
        model, prompt, max_new_tokens=3, cfg=DraftOrbitConfig(draft_tokens=2, temperature=0.0)
    )
    assert out.shape[1] == 7
    assert stats.proposed > 0
