#!/usr/bin/env python3
"""Single-device dense Transformer demo (no vLLM) — architecture='dense'."""
from __future__ import annotations

import argparse

import torch

from spider_moe.dense_model import DenseConfig, DenseLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--tokens", type=int, default=16)
    args = p.parse_args()

    cfg = DenseConfig(vocab_size=128, hidden_size=64, num_heads=4, num_layers=4, max_seq_len=128)
    model = DenseLM(cfg).to(args.device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=args.device)
    out = model.generate(prompt, max_new_tokens=args.tokens, temperature=0.0)
    print("dense generate", tuple(out.shape))
    print("architecture=dense backend=orbit-lite (no vLLM)")


if __name__ == "__main__":
    main()
