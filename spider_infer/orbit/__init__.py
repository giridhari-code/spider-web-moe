"""Orbit — Spider-native inference runtime technology.

Naming parallel (xAI-style):
  Model name:        Sentinel
  Architecture:      Spider (looped MoE + attention shells)
  Serving tech:      Orbit   ← this package  (vLLM is to dense Transformers
                                              what Orbit is to Spider)

Why not just "use vLLM"?
  vLLM optimizes *dense residual stacks* where every layer does attention
  and the dominant resource is the KV cache. Spider inverts that budget:

    • Attention lives only in prelude + coda shells (few layers).
    • The recurrent core is *loop × (router + sparse graph + sparse experts)*.
    • Early-exit can drop rows mid-orbit so they stop consuming loops.
    • KV pages are small; compute orbits are the scarce resource.

Orbit therefore treats the *active token set across recurrent loops* as the
primary scheduling unit, with paged KV as a secondary shell around it.

Core primitives
---------------
  OrbitConfig     — runtime knobs (loop budget, exit policy, page size)
  ActiveSet       — compacted batch rows still in orbit this loop
  OrbitEngine     — LLMEngine facade + scheduler orbit_work (0.2)
  draft_orbit_*  — shallow-loop draft + full-loop verify (0.4)
  orbit_step()    — one continuous-batching step with orbit accounting

vLLM concept          →  Orbit analogue
---------------------    -------------------------------------------
PagedAttention           Paged KV shells (prelude/coda only)
Continuous batching      Continuous batching (inherited)
Prefix caching           Prefix cache (inherited)
Speculative decoding     Loop-early-exit + future draft orbits
Tensor parallel          Expert-parallel inside SpiderMoELayer
"""

from .config import OrbitConfig
from .active_set import ActiveSet, compact_active, scatter_active
from .engine import OrbitEngine, OrbitStepStats
from .graphs import DecodeGraphKey, orbit_graph_allowed
from .draft import DraftOrbitConfig, DraftOrbitStats, draft_orbit_generate

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
]

__orbit_version__ = "0.4.0"
