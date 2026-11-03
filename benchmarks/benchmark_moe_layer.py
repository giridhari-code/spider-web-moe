#!/usr/bin/env python3
"""Micro-benchmark for vectorized SpiderMoELayer."""

from __future__ import annotations

import time
import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderMoELayer


def bench(device: str, steps: int = 50, warmup: int = 10):
    cfg = ModelConfig(
        vocab_size=32000,
        hidden_size=256,
        num_heads=8,
        num_experts=8,
        top_k=2,
        edges_per_expert=3,
        num_loops=4,
        expert_ffn_size=512,
        loop_embed_size=32,
        max_seq_len=256,
    ).validate()
    layer = SpiderMoELayer(cfg).to(device).eval()
    b, t, h = 4, 128, cfg.hidden_size
    x = torch.randn(b, t, h, device=device)
    x0 = torch.randn(b, t, h, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            layer(x, x0, 0)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            layer(x, x0, 0)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    tokens = b * t * steps
    print(
        f"device={device}  steps={steps}  shape=[{b},{t},{h}]  "
        f"time={elapsed:.4f}s  tokens/s={tokens / elapsed:,.0f}"
    )


if __name__ == "__main__":
    bench("cpu", steps=20, warmup=5)
    if torch.cuda.is_available():
        bench("cuda", steps=50, warmup=10)
