"""SpiderInfer — continuous-batching inference engine for Sentinel (Spider architecture)."""

from .config import CacheConfig, EngineConfig, SchedulerConfig
from .engine import LLMEngine, RequestOutput
from .sampling_params import SamplingParams
from .sequence import Sequence, SequenceStatus
from .checkpoint import load_checkpoint

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "SchedulerConfig",
    "LLMEngine",
    "RequestOutput",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "load_checkpoint",
]


# Orbit — Spider-native serving technology (vLLM analogue for Sentinel)
try:
    from .orbit import OrbitConfig, OrbitEngine, ActiveSet  # noqa: F401
except ImportError:
    pass
