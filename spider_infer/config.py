"""Inference engine configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Physical KV page pool settings."""

    page_size: int = 16
    # Total physical pages across the whole pool.
    num_pages: int = 2048
    max_requests: int = 64
    max_blocks_per_request: int = 256
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32" | "fp8" | "float8_e4m3fn"
    # FP8 scale granularity: "page" | "head" | "token" | "token_head"
    # page       — 1 scale per physical page × layer (cheapest)
    # head       — per head within a page × layer
    # token      — per token slot within a page × layer
    # token_head — per token × head (finest, most memory)
    fp8_scale_mode: str = "page"

    def torch_dtype(self):
        import torch
        from spider_moe.quant import fp8_dtype

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp8": fp8_dtype(),
            "float8_e4m3fn": fp8_dtype(),
            "float8": fp8_dtype(),
        }
        if self.dtype not in mapping:
            raise ValueError(f"unsupported cache dtype: {self.dtype}")
        return mapping[self.dtype]


@dataclass
class SchedulerConfig:
    """Continuous-batching limits (Phase 3: chunked prefill + preemption)."""

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    # Prefer decode when True.
    prefer_decode: bool = True
    # Soft memory watermark: stop admitting new prefills above this fraction.
    prefill_watermark: float = 0.90
    # Max prompt tokens processed per sequence per step (0 = unlimited).
    prefill_chunk_size: int = 256
    # Allow preempting in-progress prefills when decode needs memory.
    enable_preemption: bool = True
    # Max times a sequence may be preempted before it is aborted.
    max_preemptions: int = 3


@dataclass
class EngineConfig:
    """Top-level engine knobs."""

    cache: CacheConfig
    scheduler: SchedulerConfig
    # If None, derived from the model.
    max_model_len: int | None = None
    # Optional EOS used for stopping.
    eos_token_id: int | None = None
    # Device string passed to the model / cache.
    device: str = "cuda"
    # Seed for reproducible sampling when SamplingParams.seed is None.
    seed: int = 0
    # Capture CUDA graphs for pure decode steps (CUDA only).
    enable_cuda_graph: bool = True

    @staticmethod
    def default_for_model(max_seq_len: int = 2048, device: str = "cuda") -> "EngineConfig":
        return EngineConfig(
            cache=CacheConfig(
                page_size=16,
                num_pages=max(512, (max_seq_len // 16) * 8),
                max_requests=32,
                max_blocks_per_request=max(64, (max_seq_len + 15) // 16),
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=8,
                max_num_batched_tokens=max_seq_len,
                prefill_chunk_size=min(256, max_seq_len),
                enable_preemption=True,
            ),
            max_model_len=max_seq_len,
            device=device,
        )
