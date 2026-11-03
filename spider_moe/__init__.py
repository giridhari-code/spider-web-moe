"""Sentinel — model built on the Spider (Spider-Web Looped MoE) architecture."""
from .config import ModelConfig, TrainConfig
from .model import SpiderWebLM
from .paged_kv import PagedKVCache
from .global_paged_kv import GlobalPagedKVCache
from .branding import MODEL_NAME, MODEL_VERSION, ARCHITECTURE_NAME, display_name

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "SpiderWebLM",
    "PagedKVCache",
    "GlobalPagedKVCache",
    "MODEL_NAME",
    "MODEL_VERSION",
    "ARCHITECTURE_NAME",
    "display_name",
]
