"""Single-GPU + dense + EP helpers."""
from __future__ import annotations

import os

import torch

from spider_infer.parallel import ParallelConfig, init_parallel, is_primary, shutdown_parallel
from spider_moe.dense_model import DenseConfig, DenseLM
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.expert_parallel import ep_enabled, owns_expert


def test_dense_lm_generate():
    model = DenseLM(DenseConfig(vocab_size=64, hidden_size=32, num_heads=4, num_layers=2, ffn_size=64, max_seq_len=32)).eval()
    x = torch.randint(0, 64, (2, 5))
    out = model.generate(x, max_new_tokens=3, temperature=0.0)
    assert out.shape == (2, 8)


def test_init_parallel_single_process():
    # Ensure clean
    cfg = init_parallel("gloo")
    assert cfg.world_size() >= 1
    assert is_primary() is True
    assert ep_enabled() is False  # world 1
    shutdown_parallel()


def test_owns_expert_sharding_math():
    assert owns_expert(0, rank=0, world=2) is True
    assert owns_expert(1, rank=0, world=2) is False
    assert owns_expert(1, rank=1, world=2) is True


def test_spider_single_gpu_forward():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        y = model(x)["logits"]
    assert y.shape == (2, 6, 64)
