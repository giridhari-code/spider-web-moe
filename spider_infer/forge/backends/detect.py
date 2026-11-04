"""Probe which accelerators and kernel stacks are present."""
from __future__ import annotations

from dataclasses import dataclass

from spider_infer.forge.config import ForgeBackend


@dataclass
class BackendCapabilities:
    torch: bool = True
    cuda: bool = False
    triton: bool = False
    ascend: bool = False
    cuda_device_count: int = 0
    ascend_device_count: int = 0
    notes: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "torch": self.torch,
            "cuda": self.cuda,
            "triton": self.triton,
            "ascend": self.ascend,
            "cuda_device_count": self.cuda_device_count,
            "ascend_device_count": self.ascend_device_count,
            "notes": list(self.notes or []),
        }


def detect_capabilities() -> BackendCapabilities:
    notes: list[str] = []
    cuda = False
    cuda_count = 0
    triton = False
    ascend = False
    ascend_count = 0

    try:
        import torch

        if torch.cuda.is_available():
            cuda = True
            cuda_count = int(torch.cuda.device_count())
        # Huawei Ascend via torch_npu
        try:
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu") and torch.npu.is_available():
                ascend = True
                ascend_count = int(torch.npu.device_count())
            else:
                notes.append("torch_npu_imported_but_npu_unavailable")
        except ImportError:
            notes.append("torch_npu_not_installed")
    except ImportError:
        notes.append("torch_missing")
        return BackendCapabilities(torch=False, notes=notes)

    try:
        import triton  # noqa: F401

        triton = True
    except ImportError:
        notes.append("triton_not_installed")

    return BackendCapabilities(
        torch=True,
        cuda=cuda,
        triton=triton,
        ascend=ascend,
        cuda_device_count=cuda_count,
        ascend_device_count=ascend_count,
        notes=notes,
    )


def pick_backend(
    preferred: ForgeBackend = ForgeBackend.AUTO,
    *,
    device_hint: str = "",
    caps: BackendCapabilities | None = None,
) -> ForgeBackend:
    """Choose an executable backend; never fails — TORCH is last resort."""
    caps = caps or detect_capabilities()
    hint = (device_hint or "").lower()

    if preferred == ForgeBackend.ASCEND:
        return ForgeBackend.ASCEND if caps.ascend else (
            ForgeBackend.TORCH if not caps.cuda else ForgeBackend.TRITON if caps.triton else ForgeBackend.CUDA if caps.cuda else ForgeBackend.TORCH
        )
    if preferred == ForgeBackend.TRITON:
        return ForgeBackend.TRITON if caps.triton and caps.cuda else (
            ForgeBackend.CUDA if caps.cuda else ForgeBackend.TORCH
        )
    if preferred == ForgeBackend.CUDA:
        return ForgeBackend.CUDA if caps.cuda else ForgeBackend.TORCH
    if preferred == ForgeBackend.TORCH:
        return ForgeBackend.TORCH

    # AUTO
    if "npu" in hint or "ascend" in hint:
        if caps.ascend:
            return ForgeBackend.ASCEND
    if caps.cuda and caps.triton:
        return ForgeBackend.TRITON
    if caps.cuda:
        return ForgeBackend.CUDA
    if caps.ascend:
        return ForgeBackend.ASCEND
    return ForgeBackend.TORCH
