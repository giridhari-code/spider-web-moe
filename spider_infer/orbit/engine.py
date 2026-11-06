"""OrbitEngine — Spider-native facade over LLMEngine.

Phase 2.3: from_spiderw() for torch-free generate via NativeBackend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from spider_infer.engine import LLMEngine, RequestOutput
from spider_infer.sampling_params import SamplingParams

from .config import OrbitConfig
from spider_infer.memory.optimize import memory_report
from .dobu import dobu_stats
from spider_infer.forge import get_forge, init_forge, ForgeConfig


@dataclass
class OrbitStepStats:
    scheduled: int = 0
    is_prefill: bool = False
    tokens: int = 0
    loops_budget: int = 0
    early_exit_enabled: bool = False


@dataclass
class OrbitEngine:
    """Serving entry point branded as Orbit."""

    inner: Optional[LLMEngine] = None
    orbit_config: OrbitConfig = field(default_factory=OrbitConfig)
    _native_backend: Any = None
    _steps: int = 0
    _tokens_generated: int = 0
    _prefill_steps: int = 0
    _decode_steps: int = 0
    _last_stats: OrbitStepStats = field(default_factory=OrbitStepStats)

    @classmethod
    def from_spiderw(
        cls,
        spiderw_path: str,
        orbit_config: OrbitConfig | None = None,
        **native_kwargs: Any,
    ) -> "OrbitEngine":
        """Torch-free Orbit: NativeBackend only (no LLMEngine / no PyTorch model)."""
        from .native_backend import NativeBackend, NativeOrbitConfig

        orbit_config = (orbit_config or OrbitConfig()).validate()
        orbit_config.native_compute = True
        orbit_config.spiderw_path = spiderw_path
        ncfg = NativeOrbitConfig(
            max_loops=orbit_config.max_loops,
            early_exit=orbit_config.early_exit,
            exit_threshold=orbit_config.exit_threshold,
            max_seq_len=int(native_kwargs.get("max_seq_len", 2048)),
        )
        backend = NativeBackend.from_spiderw(spiderw_path, ncfg)
        eng = cls(inner=None, orbit_config=orbit_config, _native_backend=backend)
        return eng

    @classmethod
    def from_model(
        cls,
        model: Any,
        orbit_config: OrbitConfig | None = None,
        **engine_kwargs: Any,
    ) -> "OrbitEngine":
        orbit_config = (orbit_config or OrbitConfig()).validate()
        if hasattr(model, "cfg") and hasattr(model.cfg, "early_exit_threshold"):
            model.cfg.early_exit_threshold = orbit_config.exit_threshold
        if getattr(orbit_config, "dobu_enabled", False) and hasattr(model, "cfg"):
            model.cfg.offload_experts = True
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
                    select_top_dobu=int(getattr(orbit_config, "top_dobu", 4)),
                )
            )
        engine = LLMEngine(model, **engine_kwargs)
        if hasattr(engine.config, "scheduler"):
            engine.config.scheduler.prefer_decode = orbit_config.prefer_decode
            engine.config.scheduler.default_orbit_loops = orbit_config.max_loops
            engine.config.scheduler.max_orbit_work = orbit_config.max_orbit_work
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
        runner = getattr(engine, "model_runner", None)
        if runner is not None and hasattr(runner, "cuda_graph"):
            import torch as _torch

            cg = runner.cuda_graph
            cg.enabled = bool(orbit_config.enable_cuda_graph) and _torch.cuda.is_available()
            cg.length_bucket_size = int(orbit_config.graph_length_bucket)
        return cls(inner=engine, orbit_config=orbit_config)

    def __getattr__(self, name: str) -> Any:
        if self.inner is None:
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def model(self):
        if self.inner is None:
            return None
        return self.inner.model

    @property
    def native_backend(self):
        return self._native_backend

    def native_generate(self, prompt_ids, max_new_tokens: int = 32, eos_id=None):
        if self._native_backend is None:
            raise RuntimeError("no native backend; use OrbitEngine.from_spiderw")
        return self._native_backend.generate(
            prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id
        )

    def generate(self, *args: Any, **kwargs: Any):
        if self.inner is None:
            # map to native greedy if sampling not required
            prompt = args[0] if args else kwargs.get("prompt_ids")
            params = kwargs.get("sampling_params")
            max_tok = 32
            if params is not None and hasattr(params, "max_tokens"):
                max_tok = int(params.max_tokens)
            return self.native_generate(prompt, max_new_tokens=max_tok)
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
        if self.inner is None:
            raise RuntimeError("stream not supported on pure native OrbitEngine yet")
        if "sampling_params" not in kwargs and self.orbit_config.early_exit:
            kwargs["sampling_params"] = SamplingParams(early_exit=True)
        elif kwargs.get("sampling_params") is not None and self.orbit_config.early_exit:
            kwargs["sampling_params"].early_exit = True
        return self.inner.generate_stream(*args, **kwargs)

    def step(self) -> list:
        if self.inner is None:
            raise RuntimeError("step() requires LLMEngine; use native_generate for spiderw")
        before_running = len(self.inner.scheduler.running)
        outs = self.inner.step()
        self._steps += 1
        self._last_stats = OrbitStepStats(
            scheduled=before_running,
            loops_budget=self.orbit_config.max_loops,
            early_exit_enabled=self.orbit_config.early_exit,
        )
        for o in outs:
            self._tokens_generated += len(o.output_token_ids)
        return outs

    def orbit_snapshot(self) -> dict[str, Any]:
        snap = {
            "technology": "Orbit",
            "orbit_version": "0.5.0",
            "model": "Sentinel",
            "architecture": "Spider",
            "native_compute": bool(self.orbit_config.native_compute),
            "config": {
                "max_loops": self.orbit_config.max_loops,
                "early_exit": self.orbit_config.early_exit,
                "exit_threshold": self.orbit_config.exit_threshold,
                "prefer_decode": self.orbit_config.prefer_decode,
                "page_size": self.orbit_config.page_size,
                "enable_cuda_graph": self.orbit_config.enable_cuda_graph,
                "spiderw_path": self.orbit_config.spiderw_path,
            },
            "steps": self._steps,
            "tokens_generated": self._tokens_generated,
        }
        if self._native_backend is not None:
            snap["native"] = self._native_backend.snapshot()
        if self.inner is not None:
            snap["engine"] = self.inner.stats()
            snap["dobu"] = dobu_stats(self.inner.model)
            snap["forge"] = (
                get_forge().snapshot()
                if getattr(self.orbit_config, "forge_enabled", True)
                else None
            )
            try:
                snap["memory"] = memory_report(
                    model=self.inner.model,
                    block_manager=getattr(self.inner, "block_manager", None)
                    or getattr(
                        getattr(self.inner, "scheduler", None), "block_manager", None
                    ),
                    device=next(self.inner.model.parameters()).device,
                )
            except Exception:
                snap["memory"] = None
        return snap

    def stats(self) -> dict[str, Any]:
        return self.orbit_snapshot()
