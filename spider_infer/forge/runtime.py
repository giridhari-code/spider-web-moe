"""Forge device + synchronization primitives."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForgeDevice:
    compute: torch.device
    host: torch.device = torch.device("cpu")

    @classmethod
    def from_string(cls, device: str = "cuda") -> "ForgeDevice":
        if device.startswith("cuda") and not torch.cuda.is_available():
            return cls(compute=torch.device("cpu"))
        return cls(compute=torch.device(device))


def synchronize_forge(device: ForgeDevice | torch.device | None = None) -> None:
    """Drain compute queues before host reads (CPU↔GPU correctness)."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if isinstance(device, ForgeDevice):
        dev = device.compute
    else:
        dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
