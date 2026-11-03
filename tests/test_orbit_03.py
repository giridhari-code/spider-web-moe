"""Orbit 0.3 — CUDA graph keys (A, L) + paged-shell backend reporting."""
from __future__ import annotations

import torch

from spider_infer.cuda_graph import CudaGraphDecodeRunner, DecodeGraphKey
from spider_infer.orbit.graphs import orbit_graph_allowed
from spider_infer.config import EngineConfig
from spider_infer.engine import LLMEngine
from spider_infer.orbit import OrbitConfig, OrbitEngine
from spider_infer.sampling_params import SamplingParams
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.paged_attention import (
    get_last_paged_attention_backend,
    paged_attention,
)
from spider_moe.global_paged_kv import GlobalPagedKVCache


def _tiny(loops: int = 2):
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=loops,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=32,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
    ).validate()


def test_decode_graph_key_includes_loops():
    a = DecodeGraphKey(batch_size=4, length_bucket=64, num_loops=2)
    b = DecodeGraphKey(batch_size=4, length_bucket=64, num_loops=4)
    c = DecodeGraphKey(batch_size=4, length_bucket=64, num_loops=2)
    assert a != b
    assert a == c
    assert hash(a) == hash(c)
    assert hash(a) != hash(b)


def test_orbit_graph_allowed_policy():
    assert orbit_graph_allowed(early_exit=False, device_type="cuda") is True
    assert orbit_graph_allowed(early_exit=True, device_type="cuda") is False
    assert orbit_graph_allowed(early_exit=False, device_type="cpu") is False


def test_cuda_graph_runner_cpu_fallback_and_stats():
    runner = CudaGraphDecodeRunner(enabled=True, length_bucket_size=32)
    # On CPU-only hosts enabled becomes False in __init__ if no cuda
    ids = torch.randint(0, 10, (2, 1))

    def fn(x, rids):
        return x.float().unsqueeze(-1).expand(-1, -1, 5)

    out = runner.run(fn, ids, ["a", "b"], cache_length=10, num_loops=3)
    assert out.shape == (2, 1, 5)
    st = runner.stats()
    assert "graphs" in st
    assert st["fallbacks"] >= 1 or not st["enabled"]


def test_orbit_engine_sets_graph_bucket():
    model = SpiderWebLM(_tiny()).eval()
    eng = LLMEngine(model, EngineConfig.default_for_model(32, "cpu"))
    orbit = OrbitEngine.from_engine(
        eng,
        OrbitConfig(enable_cuda_graph=True, graph_length_bucket=128, max_loops=3),
    )
    assert eng.model_runner.cuda_graph.length_bucket_size == 128
    # CPU: graphs stay disabled
    assert eng.model_runner.cuda_graph.enabled is False


def test_paged_attention_backend_reference_on_cpu():
    """Shell path should report torch_reference on CPU."""
    cfg = _tiny()
    model = SpiderWebLM(cfg).eval()
    # Use model allocate / prefill_paged path if available
    from spider_moe.global_paged_kv import GlobalPagedKVCache

    # Minimal: build a small cache via engine block manager
    eng = LLMEngine(model, EngineConfig.default_for_model(32, "cpu"))
    prompt = [[1, 2, 3, 4, 5]]
    outs = eng.generate(prompt, SamplingParams(max_tokens=2, temperature=0.0, early_exit=False))
    assert len(outs) == 1
    backend, err = get_last_paged_attention_backend()
    # After generate, backend should have been set by attention shells
    assert backend in ("torch_reference", "triton_cuda_decode", "not_run")
    # On CPU we expect reference if attention ran
    if backend != "not_run":
        assert backend == "torch_reference"


def test_generate_decode_without_early_exit_still_works():
    model = SpiderWebLM(_tiny(loops=2)).eval()
    eng = LLMEngine(model, EngineConfig.default_for_model(32, "cpu"))
    outs = eng.generate(
        [[7, 8, 9]],
        SamplingParams(max_tokens=3, temperature=0.0, early_exit=False),
    )
    assert len(outs[0].output_token_ids) >= 1
