"""Orbit API for Lumen global workspace inspection."""
from __future__ import annotations

from typing import Any


def lumen_enabled(model) -> bool:
    rec = getattr(model, "recurrent", None)
    return rec is not None and getattr(rec, "lumen", None) is not None


def lumen_stats(model) -> dict[str, Any] | None:
    if not lumen_enabled(model):
        return None
    st = model.recurrent.lumen.stats()
    st["j_lens_attached"] = model.recurrent.lumen.j_lens is not None
    return st


def read_j_space_from_stats(stats: dict, top_k: int = 5) -> Any:
    """If forward(..., return_stats=True) included lumen fields, expose them."""
    if not stats or "lumen_slot_weights" not in stats:
        return None
    return {
        "slot_weights": stats["lumen_slot_weights"],
        "broadcast_norm": stats.get("lumen_broadcast_norm"),
        "note": "J-lens token ids require lumen(..., j_lens=True) during forward",
    }
