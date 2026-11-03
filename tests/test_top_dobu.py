"""top-dobu vocabulary (Orbit name for classic top-k chunk selection)."""
from __future__ import annotations

import torch

from spider_infer.orbit.dobu import DobuConfig, enable_dobu, dobu_stats
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def test_top_dobu_maps_to_router_width():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=8, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    enable_dobu(model, DobuConfig(top_dobu=3, max_gpu_chunks=4))
    assert model.cfg.top_k == 3
    assert model.recurrent.top_dobu == 3
    assert model.recurrent.top_k == 3
    x = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        _ = model(x)
    st = dobu_stats(model)
    assert st["top_dobu"] == 3
    assert st["top_dobu_chunks"] == 3
    assert st.get("policy") == "top-dobu"


def test_dobu_config_dict_uses_top_dobu_not_top_k_label():
    d = DobuConfig(top_dobu=2).to_dict()
    assert "top_dobu" in d
    assert "top_dobu_chunks" in d
    assert d["top_dobu_chunks"] == 2
