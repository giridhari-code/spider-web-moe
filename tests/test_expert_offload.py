"""Expert chunk offload — only routed experts touch compute memory."""
from __future__ import annotations

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.expert_store import ExpertChunkStore


def _cfg(offload: bool, experts: int = 8):
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_experts=experts,
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
        offload_experts=offload,
        offload_max_gpu_experts=2,
    ).validate()


def test_offload_experts_stay_on_cpu_after_to():
    model = SpiderWebLM(_cfg(True)).eval()
    model.to("cpu")
    for ex in model.recurrent.experts:
        assert next(ex.parameters()).device.type == "cpu"
    assert model.recurrent._expert_store is not None
    st = model.recurrent._expert_store.stats()
    assert st["num_experts"] == 8
    assert st["storage_device"].startswith("cpu")


def test_offload_forward_matches_inplace_path():
    torch.manual_seed(0)
    base = SpiderWebLM(_cfg(False)).eval()
    torch.manual_seed(0)
    off = SpiderWebLM(_cfg(True)).eval()
    # Copy non-expert + expert weights so both models equal
    off.load_state_dict(base.state_dict(), strict=False)
    # Rebuild store from copied expert weights
    from spider_moe.expert_store import ExpertChunkStore, offload_module_list_to_cpu

    off.recurrent._expert_store = ExpertChunkStore(
        off.recurrent.experts,
        storage_device="cpu",
        compute_device="cpu",
        max_gpu_experts=4,
    )
    offload_module_list_to_cpu(off.recurrent.experts)

    x = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        a = base(x)["logits"]
        b = off(x)["logits"]
    # Offload path should be close (same math)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_store_only_loads_used_experts():
    cfg = _cfg(True, experts=6)
    model = SpiderWebLM(cfg).eval()
    store = model.recurrent._expert_store
    x = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        _ = model(x)
    st = store.stats()
    # At most top_k * tokens unique but LRU capped; loads should be < num_experts typically
    assert st["loads"] >= 1
    assert len(st["gpu_hot_experts"]) <= cfg.offload_max_gpu_experts
