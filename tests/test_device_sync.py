"""CPU ↔ one-GPU synchronisation tests."""
from __future__ import annotations

import torch

from spider_infer.device_sync import DevicePlan, HybridRuntime, sync, to_compute, to_host
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def test_device_plan_cpu_only():
    plan = DevicePlan.auto(prefer_cuda=False).validate()
    assert plan.compute.type == "cpu"
    assert plan.host.type == "cpu"
    sync(plan)  # no-op


def test_to_compute_and_host_roundtrip():
    plan = DevicePlan.auto(prefer_cuda=torch.cuda.is_available()).validate()
    x = torch.arange(8, dtype=torch.long)
    y = to_compute(x, plan)
    assert y.device.type == plan.compute.type
    z = to_host(y, plan, sync_after=True)
    assert z.device.type == "cpu"
    assert torch.equal(z, x)


def test_hybrid_runtime_places_model():
    hybrid = HybridRuntime.auto(prefer_cuda=torch.cuda.is_available())
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = hybrid.place_model(SpiderWebLM(cfg).eval())
    dev = next(model.parameters()).device
    assert dev.type == hybrid.plan.compute.type
    x = hybrid.to_compute(torch.randint(0, 64, (1, 4)))
    with torch.no_grad():
        logits = model(x)["logits"]
    hybrid.sync()
    host_logits = hybrid.to_host(logits)
    assert host_logits.device.type == "cpu"
    assert torch.isfinite(host_logits).all()
    snap = hybrid.snapshot()
    assert "compute" in snap


def test_sync_after_cuda_work_is_safe():
    if not torch.cuda.is_available():
        return
    plan = DevicePlan(compute=torch.device("cuda:0"), host=torch.device("cpu")).validate()
    a = torch.randn(1000, 1000, device="cuda:0")
    b = a @ a
    sync(plan)
    # CPU can now safely read
    val = b[0, 0].item()
    assert isinstance(val, float)
