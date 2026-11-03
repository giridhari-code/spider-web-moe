#!/usr/bin/env python3
"""Multi-GPU Expert Parallel entry (Spider experts sharded).

  torchrun --nproc_per_node=2 scripts/run_orbit_ep.py

Only rank 0 binds the HTTP server; all ranks hold sharded expert weights and
participate in all_reduce during forward when requests are run on every rank.
For a first cut this script validates EP init + one collective generate on all
ranks; full data-parallel request routing is a follow-up.
"""
from __future__ import annotations

import argparse

import torch

from spider_infer.parallel import barrier, init_parallel, is_primary, shutdown_parallel
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.expert_parallel import ep_enabled, get_rank_world


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    parallel = init_parallel("nccl" if args.device.startswith("cuda") else "gloo")
    rank, world = get_rank_world()
    if is_primary():
        print(f"EP world_size={world} ep_enabled={ep_enabled()} backend={parallel.backend}")

    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_heads=4,
        num_experts=max(4, world * 2),
        top_k=2,
        edges_per_expert=2,
        num_loops=2,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=64,
        dropout=0.0,
        expert_parallel=True,
        use_rope=True,
    ).validate()

    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{int(__import__('os').environ.get('LOCAL_RANK', 0))}"
    model = SpiderWebLM(cfg).to(device).eval()

    # Collective smoke: same input on all ranks → finite logits
    x = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
    with torch.no_grad():
        out = model(x, early_exit=False)
    assert torch.isfinite(out["logits"]).all()
    barrier()
    if is_primary():
        print("EP forward OK", tuple(out["logits"].shape))
    shutdown_parallel()


if __name__ == "__main__":
    main()
