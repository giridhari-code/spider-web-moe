"""Memory optimisation helpers for Orbit KV + host/device.

Goals (single GPU + CPU host):
1. Size the paged KV pool from a byte budget, not a magic page count.
2. Prefer FP16/BF16; fall back to FP8 when the budget is tight.
3. Report live KV + CUDA allocator pressure for /v1/orbit style telemetry.
4. Optional CUDA allocator trim after large free waves.

Spider advantage: only prelude+coda attention layers hold KV, so the same
token budget costs far less HBM than a dense stack of equal depth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from spider_infer.config import CacheConfig, EngineConfig


# bytes per element
_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "float8_e4m3fn": 1,
    "float8": 1,
}


def bytes_per_kv_slot(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16",
) -> int:
    """Storage bytes for one token across all attention shells (K+V)."""
    bpe = _DTYPE_BYTES.get(dtype, 2)
    # K and V each: num_layers * num_heads * head_dim
    return 2 * num_layers * num_heads * head_dim * bpe


def estimate_pool_bytes(
    *,
    num_pages: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16",
) -> int:
    """Physical pool footprint for GlobalPagedKVCache keys+values."""
    slots = num_pages * page_size
    return slots * bytes_per_kv_slot(
        num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, dtype=dtype
    )


def available_device_bytes(device: str | torch.device = "cuda") -> int | None:
    """Free bytes on CUDA device; None on CPU."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = dev.index or 0
    free, total = torch.cuda.mem_get_info(idx)
    return int(free)


@dataclass
class MemoryPlan:
    """Resolved cache sizing for one Orbit deployment."""

    cache: CacheConfig
    pool_bytes: int
    budget_bytes: int
    dtype: str
    num_layers: int
    num_heads: int
    head_dim: int
    page_size: int
    num_pages: int
    approx_token_capacity: int
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "page_size": self.page_size,
            "num_pages": self.num_pages,
            "pool_bytes": self.pool_bytes,
            "budget_bytes": self.budget_bytes,
            "approx_token_capacity": self.approx_token_capacity,
            "notes": list(self.notes),
            "max_requests": self.cache.max_requests,
        }


def plan_kv_memory(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    budget_bytes: int | None = None,
    device: str = "cuda",
    page_size: int = 16,
    max_requests: int = 64,
    max_seq_len: int = 2048,
    prefer_dtype: str = "float16",
    min_pages: int = 64,
    headroom: float = 0.25,
) -> MemoryPlan:
    """Pick dtype + num_pages to fit inside a byte budget.

    Parameters
    ----------
    budget_bytes:
        Soft cap for the KV pool. If None and CUDA is up, uses
        ``available_device_bytes * (1 - headroom)``.
    headroom:
        Fraction of free CUDA memory left for weights / workspace.
    """
    notes: list[str] = []
    if budget_bytes is None:
        free = available_device_bytes(device)
        if free is None:
            # CPU fallback: modest default pool
            budget_bytes = 256 * 1024 * 1024
            notes.append("cpu_default_budget_256MiB")
        else:
            budget_bytes = int(free * (1.0 - headroom))
            notes.append(f"cuda_free_budget headroom={headroom}")

    dtype = prefer_dtype
    bpe = _DTYPE_BYTES.get(dtype, 2)
    slot = bytes_per_kv_slot(
        num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, dtype=dtype
    )
    # Capacity we can afford
    max_slots = max(page_size * min_pages, budget_bytes // max(slot, 1))
    num_pages = max(min_pages, max_slots // page_size)

    # Cap by max_requests * max_seq so we do not over-reserve
    need_for_full = max_requests * ((max_seq_len + page_size - 1) // page_size)
    if num_pages > need_for_full:
        notes.append("capped_to_max_requests_x_seq")
        num_pages = need_for_full

    pool = estimate_pool_bytes(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # If still over budget, step down dtype to fp8
    if pool > budget_bytes and prefer_dtype not in ("fp8", "float8_e4m3fn", "float8"):
        dtype = "fp8"
        notes.append("downgraded_dtype_to_fp8")
        slot = bytes_per_kv_slot(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, dtype=dtype
        )
        max_slots = max(page_size * min_pages, budget_bytes // max(slot, 1))
        num_pages = max(min_pages, max_slots // page_size)
        need_for_full = max_requests * ((max_seq_len + page_size - 1) // page_size)
        num_pages = min(num_pages, need_for_full)
        pool = estimate_pool_bytes(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    # Still over? shrink pages
    if pool > budget_bytes:
        slot = bytes_per_kv_slot(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, dtype=dtype
        )
        num_pages = max(min_pages, budget_bytes // max(slot * page_size, 1))
        pool = estimate_pool_bytes(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        notes.append("shrunk_num_pages_to_budget")

    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    cache = CacheConfig(
        page_size=page_size,
        num_pages=int(num_pages),
        max_requests=max_requests,
        max_blocks_per_request=max_blocks,
        dtype=dtype,
        fp8_scale_mode="page",
    )
    token_cap = int(num_pages * page_size)
    return MemoryPlan(
        cache=cache,
        pool_bytes=int(pool),
        budget_bytes=int(budget_bytes),
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_pages=int(num_pages),
        approx_token_capacity=token_cap,
        notes=notes,
    )


def engine_config_from_model_memory(
    model: torch.nn.Module,
    *,
    device: str | None = None,
    budget_bytes: int | None = None,
    max_seq_len: int | None = None,
    max_requests: int = 64,
    prefer_dtype: str = "float16",
    headroom: float = 0.25,
) -> tuple[EngineConfig, MemoryPlan]:
    """Build EngineConfig with KV pool sized to the memory plan."""
    cfg = model.cfg
    device = device or str(next(model.parameters()).device)
    num_layers = len(model._attention_layers()) if hasattr(model, "_attention_layers") else getattr(cfg, "num_prelude_layers", 1) + getattr(cfg, "num_coda_layers", 1)
    num_heads = cfg.num_heads
    head_dim = cfg.hidden_size // cfg.num_heads
    max_seq = max_seq_len or cfg.max_seq_len

    plan = plan_kv_memory(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        budget_bytes=budget_bytes,
        device=device if str(device).startswith("cuda") else "cpu",
        page_size=16,
        max_requests=max_requests,
        max_seq_len=max_seq,
        prefer_dtype=prefer_dtype,
        headroom=headroom,
    )
    eng = EngineConfig.default_for_model(max_seq, str(device))
    eng.cache = plan.cache
    eng.device = str(device)
    return eng, plan


def trim_cuda_cache(device: str | torch.device | None = None) -> None:
    """Release unused cached blocks back to the driver (best-effort)."""
    if not torch.cuda.is_available():
        return
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
            return
    torch.cuda.empty_cache()


def memory_report(
    model: torch.nn.Module | None = None,
    block_manager=None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Live snapshot for metrics / Orbit telemetry."""
    out: dict[str, Any] = {}
    if model is not None:
        n_params = sum(p.numel() for p in model.parameters())
        bytes_params = sum(p.numel() * p.element_size() for p in model.parameters())
        out["model_params"] = n_params
        out["model_param_bytes"] = bytes_params
        if hasattr(model, "_attention_layers"):
            out["attention_shell_layers"] = len(model._attention_layers())
    if block_manager is not None and hasattr(block_manager, "stats"):
        out["kv"] = block_manager.stats()
    if device is None and model is not None:
        device = next(model.parameters()).device
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            idx = dev.index or 0
            free, total = torch.cuda.mem_get_info(idx)
            out["cuda"] = {
                "device": str(dev),
                "free_bytes": int(free),
                "total_bytes": int(total),
                "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
                "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
            }
    return out
