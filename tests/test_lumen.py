"""Lumen global workspace — GWT / J-space inspired."""
from __future__ import annotations

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.lumen import LumenWorkspace
from spider_infer.orbit.lumen_api import lumen_enabled, lumen_stats


def test_lumen_module_broadcast_shape():
    w = LumenWorkspace(hidden_size=32, num_slots=4, slot_dim=8, vocab_size=50)
    h = torch.randn(2, 5, 32)
    out, state = w(h, return_state=True, j_lens=True)
    assert out.shape == h.shape
    assert state.slots.shape == (2, 5, 4, 8)
    assert state.slot_weights.shape == (2, 5, 4)
    assert state.j_logits is not None and state.j_logits.shape == (2, 5, 50)
    ids = w.read_j_space(state, top_k=3)
    assert len(ids) == 2


def test_spider_with_lumen_forward():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
        use_lumen=True, lumen_slots=4, lumen_j_lens=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    assert lumen_enabled(model)
    assert lumen_stats(model)["name"] == "Lumen"
    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 6, 64)
    with torch.no_grad():
        out2 = model(x, return_stats=True)
    # stats may be on aux path depending on forward signature
    assert out2["logits"].shape[0] == 2


def test_lumen_disabled_by_default():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    assert not lumen_enabled(model)
