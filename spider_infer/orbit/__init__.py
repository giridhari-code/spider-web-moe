"""Orbit — Spider-native inference runtime technology.

Naming parallel (xAI-style):
  Model name:        Sentinel
  Architecture:      Spider (looped MoE + attention shells)
  Serving tech:      Orbit   (vLLM is to dense Transformers
                              what Orbit is to Spider)

Versions
--------
0.1  ActiveSet + OrbitEngine facade
0.2  Shared ActiveSet in model path + scheduler orbit_work
0.3  CUDA graphs keyed by (batch, length_bucket, num_loops)
0.4  Draft orbits (shallow early-exit draft + full-loop verify)

Ready surface: OrbitEngine is a drop-in for LLMEngine (server-compatible).
"""

from .config import OrbitConfig
from .active_set import ActiveSet, compact_active, scatter_active
from .engine import OrbitEngine, OrbitStepStats
from .graphs import DecodeGraphKey, orbit_graph_allowed
from .dobu import DobuConfig, DobuStore, enable_dobu, dobu_stats
from spider_infer.forge import init_forge, get_forge, ForgeConfig, FORGE_VERSION
from .draft import (
    DraftOrbitConfig,
    DraftOrbitStats,
    draft_orbit_generate,
    draft_orbit_accept_rate_probe,
)

__all__ = [
    "OrbitConfig",
    "ActiveSet",
    "compact_active",
    "scatter_active",
    "OrbitEngine",
    "OrbitStepStats",
    "DecodeGraphKey",
    "orbit_graph_allowed",
    "DraftOrbitConfig",
    "DraftOrbitStats",
    "draft_orbit_generate",
    "DobuConfig",
    "DobuStore",
    "enable_dobu",
    "dobu_stats",
    "init_forge",
    "get_forge",
    "ForgeConfig",
    "draft_orbit_accept_rate_probe",
    "build_orbit_engine",
]

__orbit_version__ = "0.6.0"


def build_orbit_engine(model, orbit_config: OrbitConfig | None = None, **engine_kwargs):
    """Convenience: model → ready OrbitEngine (server-safe)."""
    return OrbitEngine.from_model(model, orbit_config, **engine_kwargs)
