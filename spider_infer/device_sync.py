"""One-GPU + CPU synchronisation helpers for Orbit.

Layout (typical laptop / single accelerator box)
----------------------------------------------
  CPU  — host: request I/O, tokenisation, scheduling bookkeeping, sampling RNG
  GPU0 — compute: model weights, KV pages, forward / decode

Rules
-----
1. After any CUDA work whose result the CPU will read, call ``sync()``.
2. Host→device copies of input ids use ``to_compute`` (non_blocking when pinned).
3. Device→host reads of token ids / small telemetry use ``to_host`` + ``sync``.
4. Never leave orphan CUDA streams without sync before process exit / measure.

This is not multi-GPU collectives (see ``parallel.py``); it is *host/device*
correctness for a single accelerator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DevicePlan:
    """Where Orbit runs compute vs host-side control."""

    compute: torch.device
    host: torch.device = torch.device("cpu")
    pin_memory: bool = True

    @property
    def has_cuda(self) -> bool:
        return self.compute.type == "cuda"

    @classmethod
    def auto(cls, prefer_cuda: bool = True, gpu_index: int = 0) -> "DevicePlan":
        if prefer_cuda and torch.cuda.is_available():
            return cls(compute=torch.device(f"cuda:{gpu_index}"), host=torch.device("cpu"))
        return cls(compute=torch.device("cpu"), host=torch.device("cpu"), pin_memory=False)

    def validate(self) -> "DevicePlan":
        if self.compute.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("DevicePlan.compute is CUDA but CUDA is not available")
            idx = self.compute.index or 0
            if idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"cuda:{idx} requested but only {torch.cuda.device_count()} device(s)"
                )
        if self.host.type != "cpu":
            raise ValueError("host device must be CPU for this hybrid plan")
        return self


def sync(device: torch.device | DevicePlan | None = None) -> None:
    """Block CPU until outstanding CUDA work on the plan's compute device finishes."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if isinstance(device, DevicePlan):
        dev = device.compute
    else:
        dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def to_compute(
    t: torch.Tensor,
    plan: DevicePlan,
    *,
    non_blocking: bool | None = None,
) -> torch.Tensor:
    """Move tensor to compute device (GPU0) for the forward path."""
    if t.device == plan.compute:
        return t
    nb = plan.pin_memory if non_blocking is None else non_blocking
    if plan.has_cuda and t.device.type == "cpu" and nb and plan.pin_memory:
        if not t.is_pinned():
            t = t.pin_memory()
        return t.to(plan.compute, non_blocking=True)
    return t.to(plan.compute)


def to_host(
    t: torch.Tensor,
    plan: DevicePlan,
    *,
    sync_after: bool = True,
) -> torch.Tensor:
    """Move tensor to CPU host; optional CUDA sync so values are readable."""
    if t.device == plan.host:
        return t
    out = t.to(plan.host)
    if sync_after and plan.has_cuda:
        sync(plan)
    return out


def module_to_compute(module: torch.nn.Module, plan: DevicePlan) -> torch.nn.Module:
    """Place module parameters/buffers on the compute device."""
    return module.to(plan.compute)


@dataclass
class HybridRuntime:
    """Thin handle: one GPU compute + CPU host, with explicit sync points."""

    plan: DevicePlan

    @classmethod
    def auto(cls, prefer_cuda: bool = True, gpu_index: int = 0) -> "HybridRuntime":
        return cls(plan=DevicePlan.auto(prefer_cuda=prefer_cuda, gpu_index=gpu_index).validate())

    def sync(self) -> None:
        sync(self.plan)

    def to_compute(self, t: torch.Tensor) -> torch.Tensor:
        return to_compute(t, self.plan)

    def to_host(self, t: torch.Tensor, sync_after: bool = True) -> torch.Tensor:
        return to_host(t, self.plan, sync_after=sync_after)

    def place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return module_to_compute(model, self.plan)

    def timed_sync_ms(self) -> float:
        """Host wall time to drain the compute device (0 on CPU-only)."""
        if not self.plan.has_cuda:
            return 0.0
        import time

        torch.cuda.synchronize(self.plan.compute)
        t0 = time.perf_counter()
        torch.cuda.synchronize(self.plan.compute)
        return (time.perf_counter() - t0) * 1000.0

    def snapshot(self) -> dict:
        info: dict = {
            "compute": str(self.plan.compute),
            "host": str(self.plan.host),
            "pin_memory": self.plan.pin_memory,
            "cuda_available": torch.cuda.is_available(),
        }
        if self.plan.has_cuda:
            idx = self.plan.compute.index or 0
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_index"] = idx
            mem = torch.cuda.memory_allocated(idx)
            info["memory_allocated_bytes"] = mem
        return info
