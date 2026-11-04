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
from spider_infer.memory.optimize import memory_report
from .dobu import dobu_stats
from spider_infer.forge import get_forge, init_forge, ForgeConfig


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

    Drop-in: exposes the same surface as LLMEngine via delegation so
    ``run_server(orbit)`` works without unwrapping.
    """

    inner: LLMEngine
    orbit_config: OrbitConfig
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
        # Orbit Dobu — demand-loaded expert chunks
        if getattr(orbit_config, "dobu_enabled", False) and hasattr(model, "cfg"):
            model.cfg.offload_experts = True
            # Residency only — do not touch router top_k
            top_dobu = int(
                getattr(
                    orbit_config,
                    "top_dobu",
                    getattr(orbit_config, "dobu_max_gpu_chunks", 4),
                )
            )
            model.cfg.offload_max_gpu_experts = top_dobu
        if getattr(orbit_config, "forge_enabled", True):
            init_forge(
                ForgeConfig(
                    device=str(next(model.parameters()).device),
                    select_top_k=int(getattr(model.cfg, "top_k", 2)),
                    select_top_dobu=int(
                        getattr(orbit_config, "top_dobu", 4)
                    ),
                )
            )
        engine = LLMEngine(model, **engine_kwargs)
        # Prefer decode + Orbit loop budget in scheduler
        if hasattr(engine.config, "scheduler"):
            engine.config.scheduler.prefer_decode = orbit_config.prefer_decode
            engine.config.scheduler.default_orbit_loops = orbit_config.max_loops
            engine.config.scheduler.max_orbit_work = orbit_config.max_orbit_work
        # Orbit 0.3: CUDA graph policy on the model runner
        runner = getattr(engine, "model_runner", None)
        if runner is not None and hasattr(runner, "cuda_graph"):
            import torch as _torch
            cg = runner.cuda_graph
            cg.enabled = bool(orbit_config.enable_cuda_graph) and _torch.cuda.is_available()
            cg.length_bucket_size = int(orbit_config.graph_length_bucket)
        return cls(inner=engine, orbit_config=orbit_config)

    @classmethod
    def from_engine(
        cls,
        engine: LLMEngine,
        orbit_config: OrbitConfig | None = None,
    ) -> "OrbitEngine":
        orbit_config = (orbit_config or OrbitConfig()).validate()
        if hasattr(engine.config, "scheduler"):
            engine.config.scheduler.prefer_decode = orbit_config.prefer_decode
            engine.config.scheduler.default_orbit_loops = orbit_config.max_loops
            engine.config.scheduler.max_orbit_work = orbit_config.max_orbit_work
        # Orbit 0.3: CUDA graph policy on the model runner
        runner = getattr(engine, "model_runner", None)
        if runner is not None and hasattr(runner, "cuda_graph"):
            import torch as _torch
            cg = runner.cuda_graph
            cg.enabled = bool(orbit_config.enable_cuda_graph) and _torch.cuda.is_available()
            cg.length_bucket_size = int(orbit_config.graph_length_bucket)
        return cls(inner=engine, orbit_config=orbit_config)

    # ------------------------------------------------------------------
    # Generation API (thin wrap)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def model(self):
        return self.inner.model

    @property
    def tokenizer(self):
        return self.inner.tokenizer

    @property
    def config(self):
        """EngineConfig (inner) — use orbit_config for Orbit knobs."""
        return self.inner.config

    def generate(self, *args: Any, **kwargs: Any) -> list[RequestOutput]:
        # Stamp early_exit onto sampling params when Orbit enables it.
        # Support both positional and keyword forms used by LLMEngine.generate.
        params = kwargs.get("sampling_params")
        if params is None and len(args) >= 2:
            params = args[1]
        if params is not None and self.orbit_config.early_exit:
            if isinstance(params, SamplingParams):
                params.early_exit = True
            elif isinstance(params, list):
                for p in params:
                    p.early_exit = True
        elif params is None and self.orbit_config.early_exit:
            kwargs["sampling_params"] = SamplingParams(early_exit=True)

        return self.inner.generate(*args, **kwargs)

    def generate_stream(self, *args: Any, **kwargs: Any):
        if "sampling_params" not in kwargs and self.orbit_config.early_exit:
            kwargs["sampling_params"] = SamplingParams(early_exit=True)
        elif kwargs.get("sampling_params") is not None and self.orbit_config.early_exit:
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
            loops_budget=self.orbit_config.max_loops,
            early_exit_enabled=self.orbit_config.early_exit,
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
            "orbit_version": "0.4.0",
            "model": "Sentinel",
            "architecture": "Spider",
            "config": {
                "max_loops": self.orbit_config.max_loops,
                "early_exit": self.orbit_config.early_exit,
                "exit_threshold": self.orbit_config.exit_threshold,
                "prefer_decode": self.orbit_config.prefer_decode,
                "page_size": self.orbit_config.page_size,
                "enable_cuda_graph": self.orbit_config.enable_cuda_graph,
                "graph_length_bucket": self.orbit_config.graph_length_bucket,
                "max_orbit_work": self.orbit_config.max_orbit_work,
                "dobu_enabled": getattr(self.orbit_config, "dobu_enabled", False),
                "dobu_max_gpu_chunks": getattr(self.orbit_config, "dobu_max_gpu_chunks", 4),
                "top_dobu": getattr(self.orbit_config, "top_dobu", 2),
            },
            "steps": self._steps,
            "tokens_generated": self._tokens_generated,
            "engine": base,
            "last_step": {
                "scheduled": self._last_stats.scheduled,
                "loops_budget": self._last_stats.loops_budget,
                "early_exit_enabled": self._last_stats.early_exit_enabled,
            },
            "cuda_graph": (
                self.inner.model_runner.cuda_graph.stats()
                if hasattr(self.inner, "model_runner")
                and hasattr(self.inner.model_runner, "cuda_graph")
                else None
            ),
            "dobu": dobu_stats(self.inner.model),
            "forge": get_forge().snapshot() if getattr(self.orbit_config, "forge_enabled", True) else None,
            "memory": memory_report(
                model=self.inner.model,
                block_manager=getattr(self.inner, "block_manager", None)
                or getattr(getattr(self.inner, "scheduler", None), "block_manager", None),
                device=next(self.inner.model.parameters()).device,
            ),
        }

    def stats(self) -> dict[str, Any]:
        return self.orbit_snapshot()
