"""top_k (router) and top_dobu (Dobu residency) are independent."""
from __future__ import annotations

import torch

from spider_infer.orbit.dobu import DobuConfig, enable_dobu, dobu_stats
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def _cfg(top_k=2):
    return ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=8, top_k=top_k,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()


def test_top_k_and_top_dobu_are_independent():
    model = SpiderWebLM(_cfg(top_k=2)).eval()
    assert model.cfg.top_k == 2
    enable_dobu(model, DobuConfig(top_dobu=5))
    # Router width unchanged
    assert model.cfg.top_k == 2
    assert model.recurrent.top_k == 2
    # Dobu residency is separate
    assert model.recurrent.top_dobu == 5
    assert model.recurrent._expert_store.max_gpu_experts == 5
    st = dobu_stats(model)
    assert st["top_k"] == 2
    assert st["top_dobu"] == 5
    assert st["top_k_role"] == "router_experts_per_token"
    assert st["top_dobu_role"] == "gpu_chunk_residency_lru"


def test_dobu_does_not_rewrite_top_k_in_config_dict():
    d = DobuConfig(top_dobu=6).to_dict()
    assert d["top_dobu"] == 6
    assert "top_k" not in d  # Dobu dict is memory policy only


def test_forward_with_distinct_top_k_and_top_dobu():
    model = SpiderWebLM(_cfg(top_k=3)).eval()
    enable_dobu(model, DobuConfig(top_dobu=2))  # smaller GPU LRU than top_k is ok
    x = torch.randint(0, 64, (2, 4))
    with torch.no_grad():
        y = model(x)["logits"]
    assert y.shape == (2, 4, 64)
    assert torch.isfinite(y).all()
