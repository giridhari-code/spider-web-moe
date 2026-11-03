"""Orbit technology unit tests."""
from __future__ import annotations

import torch

from spider_infer.orbit import ActiveSet, OrbitConfig, OrbitEngine, compact_active, scatter_active
from spider_infer.orbit.active_set import ActiveSet as AS
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
        num_loops=3,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=32,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
    ).validate()


def test_orbit_config_validate():
    cfg = OrbitConfig(max_loops=4, exit_threshold=0.9).validate()
    assert cfg.early_exit is True
    try:
        OrbitConfig(exit_threshold=1.5).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected threshold validation")


def test_active_set_compact_scatter():
    x = torch.arange(12, dtype=torch.float).view(4, 3)
    mask = torch.tensor([True, False, True, False])
    indices = mask.nonzero(as_tuple=False).flatten()
    active = ActiveSet(mask=mask, indices=indices)
    compacted = compact_active(x, active)
    assert compacted.shape == (2, 3)
    assert torch.equal(compacted[0], x[0])
    assert torch.equal(compacted[1], x[2])
    restored = scatter_active(x, active, compacted * 2)
    assert torch.equal(restored[0], x[0] * 2)
    assert torch.equal(restored[1], x[1])  # untouched


def test_active_set_drop():
    active = ActiveSet.all_active(4, torch.device("cpu"))
    exited = torch.tensor([True, False, True, False])
    nxt = active.drop(exited)
    assert nxt.n_active == 2
    assert torch.equal(nxt.indices, torch.tensor([1, 3]))


def test_orbit_engine_generate_smoke():
    model = SpiderWebLM(_tiny()).eval()
    orbit = OrbitEngine.from_model(
        model,
        OrbitConfig(max_loops=3, early_exit=False, track_orbit_stats=True),
        engine_config=None,
    )
    # Override device-friendly engine defaults
    from spider_infer.config import EngineConfig
    from spider_infer.engine import LLMEngine

    eng = LLMEngine.from_config(_tiny(), EngineConfig.default_for_model(32, "cpu"), device="cpu")
    orbit = OrbitEngine.from_engine(eng, OrbitConfig(max_loops=3, early_exit=True))
    prompt = [[1, 2, 3, 4, 5]]
    from spider_infer.sampling_params import SamplingParams

    outs = orbit.generate(prompt, SamplingParams(max_tokens=4, temperature=0.0, early_exit=True))
    assert len(outs) == 1
    assert len(outs[0].output_token_ids) >= 1
    snap = orbit.orbit_snapshot()
    assert snap["technology"] == "Orbit"
    assert snap["model"] == "Sentinel"
    assert snap["architecture"] == "Spider"
    assert "config" in snap
