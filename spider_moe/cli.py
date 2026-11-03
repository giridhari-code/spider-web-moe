import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from .config import ModelConfig, load_config
from .model import SpiderWebLM
from .trainer import train, train_pipeline


def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser(
        prog="spider",
        description="Sentinel 1.0 — Spider-Web Looped MoE (architecture: Spider)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--config", default="configs/train-pretrain.yaml")

    pipe = sub.add_parser("train-pipeline")
    pipe.add_argument("--config", default="configs/train-pipeline.yaml")
    pipe.add_argument("--device", default="auto")
    pipe.add_argument("--resume", default=None)

    inf = sub.add_parser("infer", help="Generate with SpiderInfer continuous-batching engine")
    inf.add_argument("--checkpoint", required=True)
    inf.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (optional)")
    inf.add_argument("--prompt", default="The future of AI is")
    inf.add_argument("--max-new-tokens", type=int, default=80)
    inf.add_argument("--temperature", type=float, default=0.8)
    inf.add_argument("--top-p", type=float, default=0.9)
    inf.add_argument("--top-k", type=int, default=0)
    inf.add_argument("--seed", type=int, default=None)
    inf.add_argument("--early-exit", action="store_true", help="Opt-in early exit (untrained by default)")
    inf.add_argument("--device", default="auto")
    inf.add_argument("--stream", action="store_true")

    serve = sub.add_parser("serve-batch", help="Batch prompts from a text file (one per line)")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--tokenizer", default=None)
    serve.add_argument("--prompts-file", required=True)
    serve.add_argument("--max-new-tokens", type=int, default=64)
    serve.add_argument("--temperature", type=float, default=0.8)
    serve.add_argument("--top-p", type=float, default=0.9)
    serve.add_argument("--device", default="auto")


    srv = sub.add_parser("serve", help="HTTP inference server (OpenAI-style)")
    srv.add_argument("--checkpoint", required=True)
    srv.add_argument("--tokenizer", default=None)
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--max-inflight", type=int, default=8)
    srv.add_argument("--default-max-tokens", type=int, default=64)
    srv.add_argument("--hard-max-tokens", type=int, default=512)
    srv.add_argument("--device", default="auto")
    srv.add_argument("--fp8-kv", action="store_true")
    srv.add_argument("--int4-moe", action="store_true")

    bench = sub.add_parser("bench")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--seq-len", type=int, default=128)
    bench.add_argument("--batch-size", type=int, default=1)

    args = p.parse_args()

    if args.command == "train":
        cfg, tc = load_config(args.config)
        model = SpiderWebLM(cfg)
        train(model, cfg, tc, device_auto())
        return

    if args.command == "train-pipeline":
        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        cfg = ModelConfig(**raw.get("model", {})).validate()
        model = SpiderWebLM(cfg)
        train_pipeline(model, cfg.__dict__, raw, args.device, args.resume)
        return

    if args.command in ("infer", "serve-batch"):
        from spider_infer import EngineConfig, LLMEngine, SamplingParams

        device = device_auto() if args.device == "auto" else args.device
        engine = LLMEngine.from_checkpoint(
            args.checkpoint,
            engine_config=EngineConfig.default_for_model(device=device),
            device=device,
            tokenizer_path=args.tokenizer,
        )
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=getattr(args, "top_k", 0) or 0,
            max_tokens=args.max_new_tokens,
            early_exit=getattr(args, "early_exit", False),
            seed=getattr(args, "seed", None),
        )

        if args.command == "serve-batch":
            lines = [
                ln.strip()
                for ln in Path(args.prompts_file).read_text().splitlines()
                if ln.strip()
            ]
            if engine.tokenizer is None:
                # byte-level fallback
                token_lists = [
                    [b % engine.model.cfg.vocab_size for b in s.encode("utf-8")] or [1]
                    for s in lines
                ]
                outs = engine.generate(token_lists, params)
                for i, out in enumerate(outs):
                    print(f"=== [{i}] {lines[i][:60]}")
                    print("tokens:", out.output_token_ids)
            else:
                texts = engine.generate_text(lines, params)
                for i, (src, txt) in enumerate(zip(lines, texts)):
                    print(f"=== [{i}] {src[:60]}")
                    print(txt)
            return

        # single prompt infer
        if args.stream:
            if engine.tokenizer is not None:
                ids = engine.tokenizer.encode(args.prompt)
            else:
                ids = [b % engine.model.cfg.vocab_size for b in args.prompt.encode("utf-8")] or [1]
            print(args.prompt, end="", flush=True)
            for tid in engine.generate_stream(ids, params):
                if engine.tokenizer is not None:
                    print(engine.tokenizer.decode([tid]), end="", flush=True)
                else:
                    print(f" {tid}", end="", flush=True)
            print()
            return

        if engine.tokenizer is not None:
            texts = engine.generate_text(args.prompt, params)
            print(texts[0])
        else:
            ids = [b % engine.model.cfg.vocab_size for b in args.prompt.encode("utf-8")] or [1]
            outs = engine.generate([ids], params)
            print("Generated token IDs:", outs[0].token_ids)
        return

    if args.command == "serve":
        from spider_infer import EngineConfig, LLMEngine
        from spider_infer.server import ServeConfig, run_server

        device = device_auto() if args.device == "auto" else args.device
        eng_cfg = EngineConfig.default_for_model(device=device)
        if args.fp8_kv:
            eng_cfg.cache.dtype = "fp8"
        engine = LLMEngine.from_checkpoint(
            args.checkpoint,
            engine_config=eng_cfg,
            device=device,
            tokenizer_path=args.tokenizer,
        )
        if args.int4_moe:
            engine.model.quantize_moe_experts()
        run_server(
            engine,
            ServeConfig(
                host=args.host,
                port=args.port,
                max_inflight=args.max_inflight,
                default_max_tokens=args.default_max_tokens,
                hard_max_tokens=args.hard_max_tokens,
            ),
        )
        return

    # bench
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = SpiderWebLM(cfg)
    model.load_state_dict(ckpt["model"])
    device = device_auto()
    model.to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "device": device,
                "seconds": elapsed,
                "tokens": args.batch_size * args.seq_len,
                "tokens_per_second": args.batch_size * args.seq_len / elapsed,
                "logits_shape": list(y["logits"].shape),
            }
        )
    )


if __name__ == "__main__":
    main()
