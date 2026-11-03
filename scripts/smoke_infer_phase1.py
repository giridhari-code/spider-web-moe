#!/usr/bin/env python3
"""Phase-1 SpiderInfer smoke test (CPU-friendly, tiny model)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running from repo root without install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_infer import EngineConfig, LLMEngine, SamplingParams


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

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
        max_seq_len=64,
    ).validate()

    model = SpiderWebLM(cfg).to(device)
    model.eval()

    engine_cfg = EngineConfig.default_for_model(max_seq_len=64, device=device)
    engine_cfg.cache.num_pages = 128
    engine_cfg.scheduler.max_num_seqs = 4
    engine = LLMEngine(model, engine_cfg)

    prompts = [
        [1, 2, 3, 4, 5],
        [10, 11, 12],
    ]
    params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=8,
        seed=42,
    )

    print("Running batch generate...")
    outputs = engine.generate(prompts, params)
    for i, out in enumerate(outputs):
        print(
            f"  req={out.request_id} "
            f"prompt={out.prompt_token_ids} "
            f"output={out.output_token_ids}"
        )

    print("Running stream generate...")
    stream_tokens = list(
        engine.generate_stream([20, 21, 22], SamplingParams(max_tokens=5, temperature=0.0))
    )
    print(f"  stream tokens={stream_tokens}")

    print("stats:", engine.stats())
    print("Phase-1 smoke test OK")


if __name__ == "__main__":
    main()
