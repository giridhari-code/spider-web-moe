"""Compare full-context recomputation with incremental KV-cached decode.

Example on Colab T4:
  python benchmarks/benchmark_incremental_decode.py --device cuda --dtype fp16

For a smaller smoke run:
  python benchmarks/benchmark_incremental_decode.py --device cpu --prompt-len 16 --decode-tokens 8
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    cfg = ModelConfig(
        vocab_size=4096, hidden_size=256, num_heads=8,
        num_experts=8, top_k=2, edges_per_expert=3, num_loops=6,
        num_prelude_layers=2, num_coda_layers=2,
        expert_ffn_size=512, loop_embed_size=32,
        max_seq_len=max(512, args.prompt_len + args.decode_tokens + 1),
        dropout=0.0,
    )
    model = SpiderWebLM(cfg).to(device=device, dtype=dtype).eval()
    tokens = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.prompt_len + args.decode_tokens),
        device=device,
    )

    @torch.inference_mode()
    def full_context_run():
        for index in range(args.decode_tokens):
            context = tokens[:, :args.prompt_len + index]
            _ = model(context, early_exit=False)["logits"][:, -1]

    @torch.inference_mode()
    def cached_run():
        prompt = tokens[:, :args.prompt_len]
        _, cache = model.prefill(prompt)
        for index in range(args.decode_tokens):
            _ , cache = model.decode_step(
                tokens[:, args.prompt_len + index:args.prompt_len + index + 1],
                cache, early_exit=False,
            )

    # Correctness check for first incremental step.
    with torch.inference_mode():
        prompt = tokens[:, :args.prompt_len]
        _, cache = model.prefill(prompt)
        cached_logits, _ = model.decode_step(
            tokens[:, args.prompt_len:args.prompt_len + 1], cache, early_exit=False
        )
        full_logits = model(
            tokens[:, :args.prompt_len + 1], early_exit=False
        )["logits"][:, -1:, :]
        torch.testing.assert_close(cached_logits, full_logits, rtol=3e-2, atol=3e-2)

    for _ in range(args.warmup):
        full_context_run()
        cached_run()
    sync(device)

    timings = {}
    for name, fn in [("full_context", full_context_run), ("incremental_kv", cached_run)]:
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        timings[name] = time.perf_counter() - start

    full_ms = timings["full_context"] * 1000 / args.decode_tokens
    cached_ms = timings["incremental_kv"] * 1000 / args.decode_tokens
    print(f"device={device} dtype={args.dtype} batch={args.batch_size}")
    print(f"prompt_len={args.prompt_len} decode_tokens={args.decode_tokens}")
    print(f"full-context: {full_ms:.3f} ms/token")
    print(f"incremental KV: {cached_ms:.3f} ms/token")
    print(f"speedup: {full_ms / cached_ms:.2f}x")


if __name__ == "__main__":
    main()
