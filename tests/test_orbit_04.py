"""Orbit 0.4 — draft orbit speculation tests."""
from __future__ import annotations

import torch

from spider_infer.orbit.draft import (
    DraftOrbitConfig,
    draft_orbit_generate,
    draft_orbit_accept_rate_probe,
)
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def _tiny(loops: int = 4):
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=loops,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=64,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
        early_exit_threshold=0.85,
    ).validate()


def test_draft_orbit_generates_requested_tokens():
    torch.manual_seed(0)
    model = SpiderWebLM(_tiny()).eval()
    prompt = torch.randint(0, 64, (1, 6))
    out, stats = draft_orbit_generate(
        model,
        prompt,
        max_new_tokens=8,
        cfg=DraftOrbitConfig(draft_tokens=4, temperature=0.0),
    )
    assert out.shape == (1, 14)
    assert stats.rounds >= 1
    assert stats.proposed >= 1
    assert stats.accepted >= 1
    assert 0.0 <= stats.accept_rate <= 1.0


def test_draft_orbit_matches_greedy_full_when_exit_never_fires():
    """If exit_head never triggers, draft==full and accept rate is 1.0."""
    torch.manual_seed(1)
    model = SpiderWebLM(_tiny(loops=3)).eval()
    with torch.no_grad():
        # Strongly discourage exit
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.fill_(-20.0)

    prompt = torch.randint(0, 64, (1, 5))
    out_spec, stats = draft_orbit_generate(
        model,
        prompt,
        max_new_tokens=6,
        cfg=DraftOrbitConfig(draft_tokens=3, temperature=0.0),
    )
    # Baseline greedy without speculation
    with torch.no_grad():
        base = model.generate(prompt, max_new_tokens=6, temperature=0.0, early_exit=False)

    assert out_spec.shape == base.shape
    assert torch.equal(out_spec, base)
    assert stats.accept_rate == 1.0


def test_draft_orbit_still_correct_when_exit_always_fires():
    """Forced early exit: output must still match full greedy verify path length."""
    torch.manual_seed(2)
    model = SpiderWebLM(_tiny(loops=4)).eval()
    with torch.no_grad():
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.fill_(20.0)

    prompt = torch.randint(0, 64, (1, 4))
    out, stats = draft_orbit_generate(
        model,
        prompt,
        max_new_tokens=5,
        cfg=DraftOrbitConfig(draft_tokens=2, temperature=0.0),
    )
    assert out.shape[1] == prompt.shape[1] + 5
    assert stats.proposed >= 5
    # Accept rate may be < 1 when draft and full disagree
    assert stats.accepted >= 5  # every position yields some accepted token


def test_accept_rate_probe_runs():
    torch.manual_seed(3)
    model = SpiderWebLM(_tiny()).eval()
    prompt = torch.randint(0, 64, (1, 4))
    stats = draft_orbit_accept_rate_probe(model, prompt, draft_tokens=2, rounds=3)
    assert stats.rounds >= 1
    assert stats.proposed > 0
