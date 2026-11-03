"""Compare contiguous incremental KV with global paged KV.

Example:
  python benchmarks/benchmark_paged_cache.py --device cuda --dtype fp16
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.paged_attention import TRITON_AVAILABLE, get_last_paged_attention_backend


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
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
    ids = torch.randint(
        0, cfg.vocab_size,
        (args.batch_size, args.prompt_len + args.decode_tokens), device=device
    )
    request_ids = [f"bench-{i}" for i in range(args.batch_size)]

    @torch.inference_mode()
    def run_contiguous():
        _, cache = model.prefill(ids[:, :args.prompt_len])
        logits = None
        for i in range(args.decode_tokens):
            logits, cache = model.decode_step(
                ids[:, args.prompt_len+i:args.prompt_len+i+1], cache, early_exit=False
            )
        return logits

    @torch.inference_mode()
    def run_paged():
        cache = model.allocate_global_paged_cache(
            num_pages=args.num_pages, page_size=args.page_size,
            max_requests=max(8, args.batch_size),
            dtype=dtype, device=device,
        )
        _, cache = model.prefill_paged(ids[:, :args.prompt_len], cache, request_ids)
        logits = None
        for i in range(args.decode_tokens):
            logits, cache = model.decode_step_paged(
                ids[:, args.prompt_len+i:args.prompt_len+i+1],
                cache, request_ids, early_exit=False,
            )
        return logits

    with torch.inference_mode():
        base = run_contiguous()
        paged = run_paged()
        torch.testing.assert_close(paged, base, rtol=3e-2, atol=3e-2)

    for _ in range(args.warmup):
        run_contiguous()
        run_paged()
    timings = {}
    for name, fn in [("contiguous", run_contiguous), ("global_paged", run_paged)]:
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        timings[name] = time.perf_counter() - start

    print(f"device={device} dtype={args.dtype} batch={args.batch_size}")
    print(f"prompt={args.prompt_len} decode={args.decode_tokens} page_size={args.page_size}")
    print(f"triton_available={TRITON_AVAILABLE}")
    print(f"last_paged_attention_backend={get_last_paged_attention_backend()}")
    for name, elapsed in timings.items():
        print(f"{name}: {elapsed*1000/args.decode_tokens:.3f} ms/decode-token")
    print(f"paged_vs_contiguous_speedup: {timings['contiguous']/timings['global_paged']:.2f}x")


if __name__ == "__main__":
    main()
