#!/usr/bin/env python3
"""Single-GPU (or CPU) Orbit serve entry — Sentinel / Spider default."""
from __future__ import annotations

import argparse

import torch

from spider_infer.config import EngineConfig
from spider_infer.orbit import OrbitConfig, build_orbit_engine
from spider_infer.device_sync import HybridRuntime
from spider_infer.server import ServeConfig, run_server
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def main():
    p = argparse.ArgumentParser(description="Orbit single-device serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-loops", type=int, default=4)
    p.add_argument("--early-exit", action="store_true")
    args = p.parse_args()

    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_heads=4,
        num_experts=8,
        top_k=2,
        edges_per_expert=2,
        num_loops=args.max_loops,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=128,
        loop_embed_size=16,
        max_seq_len=256,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
    ).validate()
    hybrid = HybridRuntime.auto(
        prefer_cuda=args.device.startswith("cuda"),
        gpu_index=int(args.device.split(":")[1]) if ":" in args.device else 0,
    )
    print("device plan", hybrid.snapshot())
    model = hybrid.place_model(SpiderWebLM(cfg).eval())
    device = str(hybrid.plan.compute)
    orbit = build_orbit_engine(
        model,
        OrbitConfig(
            max_loops=args.max_loops,
            early_exit=args.early_exit,
            enable_cuda_graph=(args.device.startswith("cuda")),
        ),
        engine_config=EngineConfig.default_for_model(cfg.max_seq_len, device),
    )
    run_server(orbit, ServeConfig(host=args.host, port=args.port, model_name="sentinel"))


if __name__ == "__main__":
    main()
