"""Orbit Dobu — demand-loaded expert weight chunks."""
from __future__ import annotations

import torch

from spider_infer.orbit.dobu import DobuConfig, enable_dobu, dobu_stats
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def _cfg(offload=False):
    return ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=6, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
        offload_experts=offload, offload_max_gpu_experts=2,
    ).validate()


def test_dobu_config_apply():
    cfg = _cfg(False)
    DobuConfig(enabled=True, top_dobu=3).apply_to_model_config(cfg)
    assert cfg.offload_experts is True
    assert cfg.offload_max_gpu_experts == 3
    assert cfg.top_k == 2  # router unchanged by Dobu


def test_enable_dobu_on_model():
    model = SpiderWebLM(_cfg(False)).eval()
    store = enable_dobu(model, DobuConfig(enabled=True, top_dobu=2))
    assert model.recurrent.offload_experts is True
    assert model.recurrent._expert_store is store
    x = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        y = model(x)["logits"]
    assert torch.isfinite(y).all()
    st = dobu_stats(model)
    assert st is not None
    assert st["name"] == "Orbit Dobu"
    assert st["loads"] >= 1


def test_dobu_parity_with_resident_experts():
    torch.manual_seed(0)
    a = SpiderWebLM(_cfg(False)).eval()
    torch.manual_seed(0)
    b = SpiderWebLM(_cfg(False)).eval()
    b.load_state_dict(a.state_dict())
    enable_dobu(b, DobuConfig(top_dobu=4))
    x = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        assert torch.allclose(a(x)["logits"], b(x)["logits"], rtol=1e-4, atol=1e-4)
