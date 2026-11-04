"""Forge device + synchronization primitives (multi-backend)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForgeDevice:
    compute: torch.device
    host: torch.device = torch.device("cpu")

    @classmethod
    def from_string(cls, device: str = "cuda") -> "ForgeDevice":
        d = (device or "cpu").lower()
        if d.startswith("npu"):
            try:
                import torch_npu  # noqa: F401

                if hasattr(torch, "npu") and torch.npu.is_available():
                    return cls(compute=torch.device(device if ":" in d else "npu:0"))
            except ImportError:
                pass
            return cls(compute=torch.device("cpu"))
        if d.startswith("cuda"):
            if not torch.cuda.is_available():
                return cls(compute=torch.device("cpu"))
            return cls(compute=torch.device(device if ":" in d else "cuda:0"))
        return cls(compute=torch.device("cpu"))


def synchronize_forge(device: ForgeDevice | torch.device | None = None) -> None:
    """Drain compute queues before host reads."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
        except Exception:
            pass
        return
    if isinstance(device, ForgeDevice):
        dev = device.compute
    else:
        dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "npu":
        try:
            torch.npu.synchronize()
        except Exception:
            pass
