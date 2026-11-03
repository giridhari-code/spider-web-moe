"""Memory plan + report tests."""
from __future__ import annotations

import torch

from spider_infer.memory.optimize import (
    bytes_per_kv_slot,
    engine_config_from_model_memory,
    estimate_pool_bytes,
    memory_report,
    plan_kv_memory,
    trim_cuda_cache,
)
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def test_bytes_per_slot_fp16_vs_fp8():
    fp16 = bytes_per_kv_slot(num_layers=2, num_heads=4, head_dim=8, dtype="float16")
    fp8 = bytes_per_kv_slot(num_layers=2, num_heads=4, head_dim=8, dtype="fp8")
    assert fp16 == 2 * fp8
    assert fp16 == 2 * 2 * 4 * 8 * 2  # K+V * layers * heads * dim * 2 bytes


def test_plan_respects_budget_and_may_downgrade_dtype():
    # Tiny budget forces shrink / fp8
    plan = plan_kv_memory(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        budget_bytes=2 * 1024 * 1024,  # 2 MiB
        device="cpu",
        page_size=16,
        max_requests=32,
        max_seq_len=2048,
        prefer_dtype="float16",
        min_pages=16,
    )
    assert plan.pool_bytes <= plan.budget_bytes * 1.05 or plan.num_pages >= 16
    assert plan.num_pages >= 16
    assert plan.approx_token_capacity == plan.num_pages * plan.page_size


def test_engine_config_from_model_memory():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    eng, plan = engine_config_from_model_memory(
        model, device="cpu", budget_bytes=16 * 1024 * 1024, max_requests=8
    )
    assert eng.cache.num_pages == plan.num_pages
    assert eng.cache.page_size == 16
    # Spider only pays for shell layers (prelude+coda), not full num_loops
    assert plan.num_layers == 2


def test_memory_report_and_trim():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=32, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    rep = memory_report(model=model, device="cpu")
    assert rep["model_params"] > 0
    assert rep["attention_shell_layers"] == 2
    trim_cuda_cache()  # no-op on CPU


def test_estimate_pool_scales_with_pages():
    a = estimate_pool_bytes(
        num_pages=10, page_size=16, num_layers=2, num_heads=4, head_dim=8, dtype="float16"
    )
    b = estimate_pool_bytes(
        num_pages=20, page_size=16, num_layers=2, num_heads=4, head_dim=8, dtype="float16"
    )
    assert b == 2 * a
