"""Checkpoint loading helpers for SpiderInfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    strict: bool = True,
) -> tuple[SpiderWebLM, ModelConfig, dict[str, Any]]:
    """Load a SpiderWebLM checkpoint.

    Returns (model, model_config, raw_checkpoint_dict).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt or "model" not in ckpt:
        raise KeyError(
            "checkpoint must contain 'model_config' and 'model' keys "
            f"(got {list(ckpt.keys())})"
        )

    cfg = ModelConfig(**ckpt["model_config"]).validate()
    model = SpiderWebLM(cfg)
    model.load_state_dict(ckpt["model"], strict=strict)
    model.to(device).eval()
    return model, cfg, ckpt
