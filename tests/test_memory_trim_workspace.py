"""Prefix byte budget, page trim, workspace pool."""
from __future__ import annotations

import torch

from spider_infer.prefix_cache import PrefixCache
from spider_infer.memory.workspace import WorkspacePool
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer.config import EngineConfig
from spider_infer.engine import LLMEngine
from spider_infer.sampling_params import SamplingParams


def test_prefix_cache_byte_budget_evicts_lru():
    freed = []
    cache = PrefixCache(
        max_entries=10,
        max_bytes=100,
        bytes_per_token=10,
        on_evict=lambda rid: freed.append(rid),
    )
    cache.insert("a", [1, 2, 3, 4, 5], 5)  # 50 bytes
    cache.insert("b", [9, 8, 7, 6, 5], 5)  # 50 bytes → total 100
    assert cache.total_bytes == 100
    cache.insert("c", [1, 1, 1, 1, 1], 5)  # needs 50 → evict oldest a
    assert "a" in freed or cache.evictions >= 1
    assert len(cache) <= 2
    assert cache.total_bytes <= 100


def test_prefix_cache_max_entries_still_works():
    cache = PrefixCache(max_entries=2, max_bytes=None, bytes_per_token=1)
    cache.insert("a", [1], 1)
    cache.insert("b", [2], 1)
    cache.insert("c", [3], 1)
    assert len(cache) == 2
    assert cache.evictions >= 1


def test_trim_to_length_releases_pages():
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, num_heads=4, num_experts=4, top_k=2,
        edges_per_expert=2, num_loops=2, num_prelude_layers=1, num_coda_layers=1,
        expert_ffn_size=64, loop_embed_size=8, max_seq_len=64, dropout=0.0,
        expert_parallel=False, use_rope=True,
    ).validate()
    model = SpiderWebLM(cfg).eval()
    eng = LLMEngine(model, EngineConfig.default_for_model(64, "cpu"), enable_prefix_cache=True)
    # Allocate a request with longer reservation via generate
    outs = eng.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, temperature=0.0))
    assert len(outs) == 1
    # Direct cache trim API
    cache = eng.block_manager.cache
    rid = "trim-test"
    cache.allocate(rid, seq_len=32)
    # simulate written length
    cache._request(rid).length = 32
    # force pages for 32 tokens
    cache.reserve(rid, 32)
    before = cache.stats()["used_pages"]
    released = cache.trim_to_length(rid, 8)
    after = cache.stats()["used_pages"]
    assert released >= 1
    assert after < before
    cache.free(rid)


def test_workspace_pool_reuses_buffers():
    pool = WorkspacePool(max_buffers_per_key=4)
    a = pool.acquire((4, 1), dtype=torch.long, device="cpu")
    pool.release(a)
    b = pool.acquire((4, 1), dtype=torch.long, device="cpu")
    assert pool.hits >= 1
    assert b.shape == (4, 1)
    st = pool.stats()
    assert st["hits"] >= 1
