"""SpiderInfer — continuous-batching inference + Orbit native path.

Torch-free:
  from spider_infer.orbit.engine import OrbitEngine
  OrbitEngine.from_spiderw("model.spiderw")

Torch path (optional extra):
  LLMEngine, load_checkpoint, ...
"""

from .sampling_params import SamplingParams
from .sequence import Sequence, SequenceStatus

__all__ = [
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
]

# Config is torch-light (torch_dtype is lazy)
try:
    from .config import CacheConfig, EngineConfig, SchedulerConfig  # noqa: F401

    __all__ += ["CacheConfig", "EngineConfig", "SchedulerConfig"]
except ImportError:
    pass

# Torch-backed engine — optional
try:
    from .engine import LLMEngine, RequestOutput  # noqa: F401
    from .checkpoint import load_checkpoint  # noqa: F401

    __all__ += ["LLMEngine", "RequestOutput", "load_checkpoint"]
except ImportError:
    pass

# Orbit native (engine is torch-free for from_spiderw)
try:
    from .orbit import OrbitConfig, OrbitEngine  # noqa: F401

    __all__ += ["OrbitConfig", "OrbitEngine"]
except ImportError:
    pass

try:
    from .orbit import (  # noqa: F401
        ActiveSet,
        build_orbit_engine,
        draft_orbit_generate,
        DraftOrbitConfig,
    )

    __all__ += ["ActiveSet", "build_orbit_engine", "draft_orbit_generate", "DraftOrbitConfig"]
except ImportError:
    pass

try:
    from .device_sync import DevicePlan, HybridRuntime, sync  # noqa: F401
except ImportError:
    pass
