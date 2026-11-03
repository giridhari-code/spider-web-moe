"""Sampling configuration for generation requests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request sampling controls.

    early_exit defaults to False so inference matches the trained full-loop depth.
    """

    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 128
    stop_token_ids: list[int] = field(default_factory=list)
    early_exit: bool = False
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
