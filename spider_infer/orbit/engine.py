"""OrbitEngine — Spider-native facade over LLMEngine.

Adds orbit accounting on top of continuous batching + paged KV without
replacing the existing scheduler. Future Orbit versions will fold loop
budget into the scheduler itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from spider_infer.engine import LLMEngine, RequestOutput
from spider_infer.sampling_params import SamplingParams
from spider_moe.model import SpiderWebLM

from .config import OrbitConfig


@dataclass
class OrbitStepStats:
    """Per-engine-step orbit telemetry."""

    scheduled: int = 0
    is_prefill: bool = False
    tokens: int = 0
    # Decode-path orbit stats (filled when model exposes them)
    loops_budget: int = 0
    early_exit_enabled: bool = False


@dataclass
class OrbitEngine:
    """Serving entry point branded as Orbit.

    Usage::

        orbit = OrbitEngine.from_model(model, OrbitConfig(early_exit=True))
        outs = orbit.generate(prompt_ids, SamplingParams(max_tokens=64))
        print(orbit.orbit_snapshot())
    """

    inner: LLMEngine
    config: OrbitConfig
    _steps: int = 0
    _tokens_generated: int = 0
    _prefill_steps: int = 0
    _decode_steps: int = 0
    _last_stats: OrbitStepStats = field(default_factory=OrbitStepStats)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: SpiderWebLM,
        orbit_config: OrbitConfig | None = None,
        **engine_kwargs: Any,
    ) -> "OrbitEngine":
        orbit_config = (orbit_config or OrbitConfig()).validate()
        # Align model early-exit threshold with Orbit policy
        if hasattr(model, "cfg") and hasattr(model.cfg, "early_exit_threshold"):
            model.cfg.early_exit_threshold = orbit_config.exit_threshold
        engine = LLMEngine(model, **engine_kwargs)
        # Prefer decode in scheduler when Orbit says so
        if hasattr(engine.config, "scheduler"):
            engine.config.scheduler.prefer_decode = orbit_config.prefer_decode
        return cls(inner=engine, config=orbit_config)

    @classmethod
    def from_engine(
        cls,
        engine: LLMEngine,
        orbit_config: OrbitConfig | None = None,
    ) -> "OrbitEngine":
        return cls(inner=engine, config=(orbit_config or OrbitConfig()).validate())

    # ------------------------------------------------------------------
    # Generation API (thin wrap)
    # ------------------------------------------------------------------

    def generate(self, *args: Any, **kwargs: Any) -> list[RequestOutput]:
        # Stamp early_exit onto sampling params when Orbit enables it.
        # Support both positional and keyword forms used by LLMEngine.generate.
        params = kwargs.get("sampling_params")
        if params is None and len(args) >= 2:
            params = args[1]
        if params is not None and self.config.early_exit:
            if isinstance(params, SamplingParams):
                params.early_exit = True
            elif isinstance(params, list):
                for p in params:
                    p.early_exit = True
        elif params is None and self.config.early_exit:
            kwargs["sampling_params"] = SamplingParams(early_exit=True)

        return self.inner.generate(*args, **kwargs)

    def generate_stream(self, *args: Any, **kwargs: Any):
        if "sampling_params" not in kwargs and self.config.early_exit:
            kwargs["sampling_params"] = SamplingParams(early_exit=True)
        elif kwargs.get("sampling_params") is not None and self.config.early_exit:
            kwargs["sampling_params"].early_exit = True
        return self.inner.generate_stream(*args, **kwargs)

    def step(self) -> list[RequestOutput]:
        before_running = len(self.inner.scheduler.running)
        outs = self.inner.step()
        self._steps += 1
        # Heuristic accounting from scheduler side-effects
        # (detailed loop stats live inside model forward; exposed via snapshot)
        self._last_stats = OrbitStepStats(
            scheduled=before_running,
            loops_budget=self.config.max_loops,
            early_exit_enabled=self.config.early_exit,
        )
        for o in outs:
            self._tokens_generated += len(o.output_token_ids)
        return outs

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def orbit_snapshot(self) -> dict[str, Any]:
        base = self.inner.stats()
        return {
            "technology": "Orbit",
            "orbit_version": "0.1.0",
            "model": "Sentinel",
            "architecture": "Spider",
            "config": {
                "max_loops": self.config.max_loops,
                "early_exit": self.config.early_exit,
                "exit_threshold": self.config.exit_threshold,
                "prefer_decode": self.config.prefer_decode,
                "page_size": self.config.page_size,
            },
            "steps": self._steps,
            "tokens_generated": self._tokens_generated,
            "engine": base,
            "last_step": {
                "scheduled": self._last_stats.scheduled,
                "loops_budget": self._last_stats.loops_budget,
                "early_exit_enabled": self._last_stats.early_exit_enabled,
            },
        }

    def stats(self) -> dict[str, Any]:
        return self.orbit_snapshot()
