"""Model backends for Orbit serve.

- spider: Sentinel / Spider architecture (default)
- dense:  minimal dense Transformer (single-GPU path; multi-GPU TP later)
"""
from .registry import get_backend, list_backends
from . import spider as _spider  # noqa: F401
from . import dense as _dense  # noqa: F401

__all__ = ["get_backend", "list_backends"]
