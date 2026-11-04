"""Forge 0.5 — multi-backend capability layer (Torch / Triton / CUDA / Ascend)."""
from .detect import BackendCapabilities, detect_capabilities, pick_backend
from .ascend import AscendOps, ascend_available

__all__ = [
    "BackendCapabilities",
    "detect_capabilities",
    "pick_backend",
    "AscendOps",
    "ascend_available",
]
