#!/usr/bin/env python3
"""Phase B — offline inference microbench for Sentinel (Spider architecture).

Measures:
  - Prefill throughput (tokens/s)
  - TTFT proxy (prefill latency for prompt)
  - Decode TPOT (ms/token) and tokens/s
  - Sparse expert path vs dense reference (micro)
  - Early-exit active-row compaction effect on decode loops

Usage:
  python scripts/bench_infer.py
  python scripts/bench_infer.py --device cuda --batch 8 --prompt-len 128 --gen-tokens 64
  python scripts/bench_infer.py --json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.sparse_experts import sparse_expert_delta


@dataclass
class BenchResult:
    device: str
    batch: int
    prompt_len: int
    gen_tokens: int
    num_loops: int
    prefill_ms: float
    prefill_tok_s: float
    decode_ms_per_token: float
    decode_tok_s: float
    end_to_end_ms: float
    end_to_end_tok_s: float
    sparse_vs_dense_max_abs: float
    sparse_speedup: float | None
    early_exit_decode_ms_per_token: float | None
    early_exit_speedup: float | None


def _cfg(
    *,
    hidden: int = 64,
    heads: int = 4,
    experts: int = 8,
    top_k: int = 2,
    loops: int = 4,
    max_seq: int = 256,
    vocab: int = 256,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_heads=heads,
        num_experts=experts,
        top_k=top_k,
        edges_per_expert=2,
        num_loops=loops,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=hidden * 2,
        loop_embed_size=16,
        max_seq_len=max_seq,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
        early_exit_threshold=0.85,
    ).validate()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dense_expert_delta(messages, top_indices, top_weights, w_up, b_up, w_down, b_down):
    b, t, width = messages.shape
    e = w_up.shape[0]
    up = torch.einsum("bth,eoh->bteo", messages, w_up) + b_up.view(1, 1, e, -1)
    up = F.gelu(up)
    expert_stack = torch.einsum("bteu,eou->bteo", up, w_down) + b_down.view(1, 1, e, -1)
    expert_weight = messages.new_zeros(b, t, e)
    expert_weight.scatter_add_(-1, top_indices, top_weights)
    return (expert_stack * expert_weight.unsqueeze(-1)).sum(dim=2)


def bench_sparse_micro(device: torch.device, rounds: int = 20) -> tuple[float, float | None]:
    """Return (max_abs_diff, speedup sparse vs dense). Speedup None on tiny CPU noise."""
    torch.manual_seed(0)
    b, t, h, e, k, h_up = 8, 32, 64, 8, 2, 128
    messages = torch.randn(b, t, h, device=device)
    scores = torch.randn(b, t, e, device=device)
    top_weights, top_indices = scores.softmax(-1).topk(k, dim=-1)
    top_weights = top_weights / top_weights.sum(-1, keepdim=True).clamp_min(1e-9)
    w_up = torch.randn(e, h_up, h, device=device)
    b_up = torch.randn(e, h_up, device=device)
    w_down = torch.randn(e, h, h_up, device=device)
    b_down = torch.randn(e, h, device=device)

    sparse = sparse_expert_delta(
        messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
        num_experts=e, top_k=k, expert_parallel=False,
    )
    dense = _dense_expert_delta(
        messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
    )
    max_abs = (sparse - dense).abs().max().item()

    # Warmup
    for _ in range(5):
        sparse_expert_delta(
            messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
            num_experts=e, top_k=k, expert_parallel=False,
        )
        _dense_expert_delta(
            messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
        )
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(rounds):
        sparse_expert_delta(
            messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
            num_experts=e, top_k=k, expert_parallel=False,
        )
    _sync(device)
    sparse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(rounds):
        _dense_expert_delta(
            messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
        )
    _sync(device)
    dense_s = time.perf_counter() - t0

    speedup = dense_s / sparse_s if sparse_s > 0 else None
    return max_abs, speedup


def bench_model(
    *,
    device: torch.device,
    batch: int,
    prompt_len: int,
    gen_tokens: int,
    loops: int,
    warmup: int = 2,
    repeats: int = 5,
) -> BenchResult:
    cfg = _cfg(loops=loops, max_seq=max(256, prompt_len + gen_tokens + 8))
    model = SpiderWebLM(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            logits, cache = model.prefill(prompt)
            tok = logits[:, -1, :].argmax(-1, keepdim=True)
            for _ in range(min(4, gen_tokens)):
                logits, cache = model.decode_step(tok, cache, early_exit=False)
                tok = logits[:, -1, :].argmax(-1, keepdim=True)
    _sync(device)

    prefill_times: list[float] = []
    decode_times: list[float] = []
    e2e_times: list[float] = []

    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            logits, cache = model.prefill(prompt)
            _sync(device)
            t1 = time.perf_counter()
            prefill_times.append(t1 - t0)

            tok = logits[:, -1, :].argmax(-1, keepdim=True)
            t_dec0 = time.perf_counter()
            for _ in range(gen_tokens):
                logits, cache = model.decode_step(tok, cache, early_exit=False)
                tok = logits[:, -1, :].argmax(-1, keepdim=True)
            _sync(device)
            t_dec1 = time.perf_counter()
            decode_times.append(t_dec1 - t_dec0)
            e2e_times.append(t_dec1 - t0)

    prefill_s = statistics.median(prefill_times)
    decode_s = statistics.median(decode_times)
    e2e_s = statistics.median(e2e_times)

    # Early-exit decode: force exit_head to exit after loop 0 for all rows
    early_ms = None
    early_speedup = None
    if gen_tokens > 0:
        with torch.no_grad():
            model.recurrent.exit_head.weight.zero_()
            model.recurrent.exit_head.bias.fill_(20.0)
            logits, cache = model.prefill(prompt)
            tok = logits[:, -1, :].argmax(-1, keepdim=True)
            _sync(device)
            times = []
            for _ in range(repeats):
                # re-prefill for clean cache
                logits, cache = model.prefill(prompt)
                tok = logits[:, -1, :].argmax(-1, keepdim=True)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(gen_tokens):
                    logits, cache = model.decode_step(tok, cache, early_exit=True)
                    tok = logits[:, -1, :].argmax(-1, keepdim=True)
                _sync(device)
                times.append(time.perf_counter() - t0)
            early_s = statistics.median(times)
            early_ms = (early_s / gen_tokens) * 1000
            if decode_s > 0:
                early_speedup = decode_s / early_s

    max_abs, sparse_speedup = bench_sparse_micro(device)

    return BenchResult(
        device=str(device),
        batch=batch,
        prompt_len=prompt_len,
        gen_tokens=gen_tokens,
        num_loops=loops,
        prefill_ms=prefill_s * 1000,
        prefill_tok_s=(batch * prompt_len) / prefill_s if prefill_s > 0 else 0.0,
        decode_ms_per_token=(decode_s / gen_tokens) * 1000 if gen_tokens else 0.0,
        decode_tok_s=(batch * gen_tokens) / decode_s if decode_s > 0 else 0.0,
        end_to_end_ms=e2e_s * 1000,
        end_to_end_tok_s=(batch * gen_tokens) / e2e_s if e2e_s > 0 else 0.0,
        sparse_vs_dense_max_abs=max_abs,
        sparse_speedup=sparse_speedup,
        early_exit_decode_ms_per_token=early_ms,
        early_exit_speedup=early_speedup,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Sentinel inference microbench (Phase B)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    result = bench_model(
        device=device,
        batch=args.batch,
        prompt_len=args.prompt_len,
        gen_tokens=args.gen_tokens,
        loops=args.loops,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    data = asdict(result)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    print("=== Sentinel / Spider inference bench (Phase B) ===")
    print(f"device={result.device}  batch={result.batch}  "
          f"prompt={result.prompt_len}  gen={result.gen_tokens}  loops={result.num_loops}")
    print(f"prefill:     {result.prefill_ms:.2f} ms   {result.prefill_tok_s:.1f} tok/s")
    print(f"decode:      {result.decode_ms_per_token:.3f} ms/tok   {result.decode_tok_s:.1f} tok/s")
    print(f"end-to-end:  {result.end_to_end_ms:.2f} ms   {result.end_to_end_tok_s:.1f} gen-tok/s")
    print(f"sparse vs dense max|Δ|: {result.sparse_vs_dense_max_abs:.2e}")
    if result.sparse_speedup is not None:
        print(f"sparse speedup (micro): {result.sparse_speedup:.2f}x")
    if result.early_exit_decode_ms_per_token is not None:
        print(f"early-exit decode: {result.early_exit_decode_ms_per_token:.3f} ms/tok  "
              f"speedup={result.early_exit_speedup:.2f}x (forced exit after loop 0)")


if __name__ == "__main__":
    main()
