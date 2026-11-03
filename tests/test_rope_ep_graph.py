"""RoPE, expert-parallel helpers, CUDA graph smoke tests."""

from __future__ import annotations

import torch
import pytest

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.rope import apply_rope, precompute_freqs_cis
from spider_moe.expert_parallel import ep_enabled, owns_expert, get_rank_world
from spider_infer.cuda_graph import CudaGraphDecodeRunner


def test_rope_shapes_and_rotation():
    cos, sin = precompute_freqs_cis(16, 32)
    assert cos.shape == (32, 16)
    x = torch.randn(2, 4, 8, 16)
    y = apply_rope(x, cos, sin, position=0)
    assert y.shape == x.shape
    # Different position should change values
    y2 = apply_rope(x, cos, sin, position=4)
    assert not torch.allclose(y, y2)


def test_owns_expert_sharding():
    assert owns_expert(0, rank=0, world=2)
    assert not owns_expert(1, rank=0, world=2)
    assert owns_expert(1, rank=1, world=2)
    assert not ep_enabled()  # single-process tests
    assert get_rank_world() == (0, 1)


def test_model_with_rope_forward():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=2, top_k=1,
        edges_per_expert=1, num_loops=1, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    x = torch.randint(0, 64, (2, 8))
    out = model(x)
    assert out["logits"].shape == (2, 8, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_runner_smoke():
    device = "cuda"
    runner = CudaGraphDecodeRunner(enabled=True, length_bucket_size=16)

    def fn(ids, rids):
        # trivial stand-in for decode logits
        return ids.float().sum(-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 8)

    ids = torch.ones(2, 1, dtype=torch.long, device=device)
    out1 = runner.run(fn, ids, ["a", "b"], cache_length=10)
    out2 = runner.run(fn, ids * 2, ["a", "b"], cache_length=10)
    assert out1.shape[-1] == 8
    assert out2.shape[-1] == 8
