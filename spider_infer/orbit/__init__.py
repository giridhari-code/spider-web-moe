"""Orbit — Spider-native inference runtime.

Torch-free entry:
  from spider_infer.orbit.engine import OrbitEngine
  OrbitEngine.from_spiderw(...)

Full Orbit surface (ActiveSet, draft, forge) may require torch.
"""

from .config import OrbitConfig
from .engine import OrbitEngine, OrbitStepStats

__all__ = [
    "OrbitConfig",
    "OrbitEngine",
    "OrbitStepStats",
    "build_orbit_engine",
]

__orbit_version__ = "0.6.1"

# Optional torch-backed helpers — soft import so native path works without torch
try:
    from .active_set import ActiveSet, compact_active, scatter_active  # noqa: F401

    __all__ += ["ActiveSet", "compact_active", "scatter_active"]
except ImportError:
    pass

try:
    from .graphs import DecodeGraphKey, orbit_graph_allowed  # noqa: F401

    __all__ += ["DecodeGraphKey", "orbit_graph_allowed"]
except ImportError:
    pass

try:
    from .dobu import DobuConfig, DobuStore, enable_dobu, dobu_stats  # noqa: F401

    __all__ += ["DobuConfig", "DobuStore", "enable_dobu", "dobu_stats"]
except ImportError:
    pass

try:
    from spider_infer.forge import init_forge, get_forge, ForgeConfig, FORGE_VERSION  # noqa: F401

    __all__ += ["init_forge", "get_forge", "ForgeConfig"]
except ImportError:
    pass

try:
    from .lumen_api import lumen_enabled, lumen_stats  # noqa: F401

    __all__ += ["lumen_enabled", "lumen_stats"]
except ImportError:
    pass

try:
    from .draft import (  # noqa: F401
        DraftOrbitConfig,
        DraftOrbitStats,
        draft_orbit_generate,
        draft_orbit_accept_rate_probe,
    )

    __all__ += [
        "DraftOrbitConfig",
        "DraftOrbitStats",
        "draft_orbit_generate",
        "draft_orbit_accept_rate_probe",
    ]
except ImportError:
    pass


def build_orbit_engine(model, orbit_config: OrbitConfig | None = None, **engine_kwargs):
    """Convenience: torch model → OrbitEngine (requires torch)."""
    return OrbitEngine.from_model(model, orbit_config, **engine_kwargs)
