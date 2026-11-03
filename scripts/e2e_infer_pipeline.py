#!/usr/bin/env python3
"""End-to-end SpiderInfer pipeline smoke (no external checkpoint required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=2,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=128,
        loop_embed_size=16,
        max_seq_len=128,
    ).validate()
    model = SpiderWebLM(cfg).to(device).eval()

    eng_cfg = EngineConfig.default_for_model(max_seq_len=128, device=device)
    eng_cfg.cache.dtype = "float32" if device == "cpu" else "float16"
    eng_cfg.cache.num_pages = 256
    eng_cfg.scheduler.prefill_chunk_size = 16
    eng_cfg.scheduler.max_num_seqs = 4

    engine = LLMEngine(model, eng_cfg, enable_prefix_cache=True)

    prompts = [
        list(range(1, 40)),   # long → chunked prefill
        list(range(5, 20)),
        list(range(1, 40)),   # duplicate of first → prefix cache opportunity
    ]
    params = SamplingParams(max_tokens=8, temperature=0.0, seed=0)

    print("batch generate...")
    outs = engine.generate(prompts, params)
    for o in outs:
        hit = o.metadata.get("prefix_cache_hit", False)
        print(f"  {o.request_id} out={o.output_token_ids} prefix_hit={hit}")

    print("stream...")
    stream = list(engine.generate_stream(list(range(10, 18)), SamplingParams(max_tokens=4, temperature=0.0)))
    print("  stream", stream)

    print("stats", engine.stats())
    print("E2E pipeline OK")


if __name__ == "__main__":
    main()
