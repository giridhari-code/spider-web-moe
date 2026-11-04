"""Forge fabric — all five pillars present."""
from __future__ import annotations

import torch

from spider_infer.forge import (
    FORGE_VERSION,
    ForgeConfig,
    ForgeBackend,
    init_forge,
    get_forge,
)
from spider_infer.forge.architecture import FORGE_PILLARS, DEEPINFER_MAP


def test_all_pillars_in_architecture():
    assert set(FORGE_PILLARS) == {"attention", "gemm", "ep", "tile", "select"}
    assert set(DEEPINFER_MAP.keys()) == {
        "FlashMLA", "DeepGEMM", "DeepEP", "TileLang", "DeepSelect"
    }


def test_init_forge_snapshot_complete():
    rt = init_forge(ForgeConfig(device="cpu", backend=ForgeBackend.TORCH))
    snap = rt.snapshot()
    assert snap["technology"] == "Forge"
    for p in FORGE_PILLARS:
        assert p in snap["pillars"]
        assert snap["pillars"][p]["analogue"] in DEEPINFER_MAP


def test_select_top_k_vs_top_dobu_independent():
    rt = init_forge(ForgeConfig(select_top_k=2, select_top_dobu=5, device="cpu"))
    logits = torch.randn(3, 8)
    w, idx = rt.select.route(logits, top_k=2)
    assert idx.shape[-1] == 2
    assert rt.select.top_dobu_budget() == 5


def test_gemm_and_attention_smoke():
    rt = init_forge(ForgeConfig(device="cpu"))
    x = torch.randn(4, 16)
    w = torch.randn(32, 16)
    y = rt.gemm.linear(x, w)
    assert y.shape == (4, 32)
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    o = rt.attention.decode(q, k, v)
    assert o.shape == q.shape


def test_ep_combine_single_rank_noop():
    rt = init_forge(ForgeConfig(device="cpu"))
    d = torch.randn(2, 3, 4)
    out = rt.ep.combine(d.clone())
    assert torch.allclose(out, d)


def test_tile_registry():
    rt = init_forge(ForgeConfig(device="cpu"))
    spec = rt.tile.get_spec("expert_ffn")
    assert spec.block_m > 0
    st = rt.tile.stats()
    assert "expert_ffn" in st["specs"]
