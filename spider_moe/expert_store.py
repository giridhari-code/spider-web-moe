"""Orbit Dobu storage backend — load only experts a token routes to.

Orbit Dobu (Orbit / Spider):
  Full MoE weight tensors do not need to sit on the GPU.
  Keep expert chunks on CPU (pinned); after the router picks top-dobu
  (classic name: top-k), materialise only those expert FFNs on compute,
  then optionally drop them so idle experts never occupy VRAM.

This is the sparse-serving analogue of "don't load the whole LLM into RAM".
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ExpertChunk:
    """One expert's FFN weights (storage device, usually CPU)."""

    w_up: torch.Tensor
    b_up: torch.Tensor
    w_down: torch.Tensor
    b_down: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = True) -> "ExpertChunk":
        return ExpertChunk(
            w_up=self.w_up.to(device, non_blocking=non_blocking),
            b_up=self.b_up.to(device, non_blocking=non_blocking),
            w_down=self.w_down.to(device, non_blocking=non_blocking),
            b_down=self.b_down.to(device, non_blocking=non_blocking),
        )


class ExpertChunkStore:
    """CPU-resident expert catalogue + small GPU LRU working set.

    Parameters
    ----------
    max_gpu_experts:
        How many expert chunks may stay hot on the compute device.
        0 = always stream (minimum VRAM, more H2D).
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        *,
        storage_device: torch.device | str = "cpu",
        compute_device: torch.device | str | None = None,
        max_gpu_experts: int = 4,
        pin_memory: bool = True,
    ):
        storage_device = torch.device(storage_device)
        self.storage_device = storage_device
        self.compute_device = (
            torch.device(compute_device)
            if compute_device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_gpu_experts = max(0, int(max_gpu_experts))
        self.pin_memory = (
            bool(pin_memory)
            and storage_device.type == "cpu"
            and torch.cuda.is_available()
        )

        self._chunks: list[ExpertChunk] = []
        for expert in experts:
            # Sequential(Linear, GELU, Linear)
            up, _, down = expert[0], expert[1], expert[2]
            w_up = up.weight.detach().to(storage_device)
            b_up = up.bias.detach().to(storage_device)
            w_down = down.weight.detach().to(storage_device)
            b_down = down.bias.detach().to(storage_device)
            if self.pin_memory:
                w_up = w_up.pin_memory()
                b_up = b_up.pin_memory()
                w_down = w_down.pin_memory()
                b_down = b_down.pin_memory()
            self._chunks.append(
                ExpertChunk(w_up=w_up, b_up=b_up, w_down=w_down, b_down=b_down)
            )

        self._gpu_lru: OrderedDict[int, ExpertChunk] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.loads = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def _materialise(self, expert_id: int) -> ExpertChunk:
        if expert_id in self._gpu_lru:
            self.hits += 1
            self._gpu_lru.move_to_end(expert_id)
            return self._gpu_lru[expert_id]

        self.misses += 1
        self.loads += 1
        chunk = self._chunks[expert_id].to(self.compute_device, non_blocking=True)
        if self.max_gpu_experts > 0:
            self._gpu_lru[expert_id] = chunk
            self._gpu_lru.move_to_end(expert_id)
            while len(self._gpu_lru) > self.max_gpu_experts:
                self._gpu_lru.popitem(last=False)
        return chunk

    def release_gpu(self) -> None:
        """Drop the hot GPU working set (experts remain on storage device)."""
        self._gpu_lru.clear()
        if self.compute_device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def sparse_delta(
        self,
        messages: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        top_dobu: int | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Same contract as sparse_expert_delta, but weights come from chunks.

        ``top_dobu`` is the Orbit name for how many chunks activate per token.
        ``top_k`` remains accepted as a legacy alias.
        """
        n = top_dobu if top_dobu is not None else top_k
        if n is None:
            raise TypeError("top_dobu (or legacy top_k) is required")
        top_dobu = int(n)
        # Follow activation device (model may have been .to(cuda) after init).
        self.compute_device = messages.device
        b, t, width = messages.shape
        flat_msg = messages.reshape(-1, width)
        flat_idx = top_indices.reshape(-1, top_dobu)
        flat_w = top_weights.reshape(-1, top_dobu)
        n_tokens = flat_msg.shape[0]
        delta_flat = flat_msg.new_zeros(n_tokens, width)

        # Unique experts touched this step — only these chunks touch compute device
        used = flat_idx.unique().tolist()
        for eid in used:
            eid = int(eid)
            token_weight = flat_msg.new_zeros(n_tokens)
            for k in range(top_dobu):
                mask_k = flat_idx[:, k] == eid
                token_weight = token_weight + flat_w[:, k] * mask_k.to(flat_w.dtype)
            active = token_weight > 0
            if not active.any():
                continue
            chunk = self._materialise(eid)
            # Align dtypes with activations
            w_up = chunk.w_up.to(dtype=flat_msg.dtype)
            b_up = chunk.b_up.to(dtype=flat_msg.dtype)
            w_down = chunk.w_down.to(dtype=flat_msg.dtype)
            b_down = chunk.b_down.to(dtype=flat_msg.dtype)
            toks = flat_msg[active]
            up = F.linear(toks, w_up, b_up)
            up = F.gelu(up)
            out = F.linear(up, w_down, b_down)
            delta_flat[active] = delta_flat[active] + out * token_weight[active].unsqueeze(-1)

        return delta_flat.view(b, t, width)

    def stats(self) -> dict:
        storage_bytes = sum(
            c.w_up.numel() * c.w_up.element_size()
            + c.b_up.numel() * c.b_up.element_size()
            + c.w_down.numel() * c.w_down.element_size()
            + c.b_down.numel() * c.b_down.element_size()
            for c in self._chunks
        )
        gpu_bytes = sum(
            c.w_up.numel() * c.w_up.element_size()
            + c.b_up.numel() * c.b_up.element_size()
            + c.w_down.numel() * c.w_down.element_size()
            + c.b_down.numel() * c.b_down.element_size()
            for c in self._gpu_lru.values()
        )
        return {
            "num_experts": len(self._chunks),
            "gpu_hot_experts": list(self._gpu_lru.keys()),
            "max_gpu_experts": self.max_gpu_experts,
            "storage_device": str(self.storage_device),
            "compute_device": str(self.compute_device),
            "storage_bytes": storage_bytes,
            "gpu_working_set_bytes": gpu_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "loads": self.loads,
            "policy": "top-dobu",
        }


def offload_module_list_to_cpu(experts: nn.ModuleList) -> None:
    """Move expert nn.Modules to CPU so they no longer occupy compute VRAM."""
    for ex in experts:
        ex.to("cpu")
