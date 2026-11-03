"""Distributed helpers for Orbit multi-GPU (Expert Parallel first).

Single-GPU: init_parallel() is a no-op when WORLD_SIZE=1.
Multi-GPU:  torchrun --nproc_per_node=N ...  then init_parallel("nccl")
            Spider experts shard by expert_id % world_size (see expert_parallel.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParallelConfig:
    """Parallel plan for one Orbit deployment."""

    backend: str = "gloo"  # gloo CPU-safe; nccl for CUDA
    expert_parallel: bool = True
    # Future: tensor_parallel_size for dense backend
    tensor_parallel_size: int = 1
    architecture: str = "spider"  # "spider" | "dense"

    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", "1"))

    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", "0"))


def init_parallel(backend: str | None = None) -> ParallelConfig:
    """Initialize torch.distributed if launched under torchrun / env://."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return ParallelConfig(backend=backend or "gloo", expert_parallel=False)

    if not dist.is_initialized():
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available() and backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return ParallelConfig(
        backend=backend or "gloo",
        expert_parallel=True,
        architecture="spider",
    )


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_primary() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def shutdown_parallel() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
