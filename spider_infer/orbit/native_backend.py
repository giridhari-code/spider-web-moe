"""Orbit native compute backend — Phase 1 / 1.1.

Runs Spider MoE loop steps through libspider_kernels without torch on that path.
Phase 1.1: edge_value destination projection + graph_v2 when weights present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .weight_loader import SpiderWBundle, load_spiderw

try:
    from kernels.python.spider_native import NativeKernels, build_csr_from_edges
except ImportError:
    try:
        from spider_web_moe.kernels.python.spider_native import (  # type: ignore
            NativeKernels,
            build_csr_from_edges,
        )
    except ImportError:
        NativeKernels = None  # type: ignore
        build_csr_from_edges = None  # type: ignore


def _edge_project_np(hidden: np.ndarray, W_edge: np.ndarray) -> np.ndarray:
    """projected[b,e,:] = hidden[b] @ W_edge[e].T with W [E,H,H] as Linear [out,in]."""
    B, H = hidden.shape
    E = W_edge.shape[0]
    out = np.zeros((B, E, H), dtype=np.float32)
    for e in range(E):
        out[:, e, :] = hidden @ W_edge[e].T
    return out


def _graph_v2_np(
    projected: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_w: np.ndarray,
    assign: np.ndarray,
) -> np.ndarray:
    B, E, H = projected.shape
    K = assign.shape[1]
    messages = np.zeros((B, H), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            src = int(assign[b, k])
            if src < 0:
                continue
            for edge in range(int(row_ptr[src]), int(row_ptr[src + 1])):
                dst = int(col_idx[edge])
                messages[b] += float(edge_w[edge]) * projected[b, dst]
    return messages


@dataclass
class NativeOrbitConfig:
    max_loops: int = 4
    early_exit: bool = True
    exit_threshold: float = 0.85
    residual_scale: float = 1.0
    reinject_weight: float = 0.0
    eps: float = 1e-5
    prefer_cuda: bool = True


@dataclass
class NativeMoEState:
    h: np.ndarray
    x0: np.ndarray


@dataclass
class NativeBackend:
    bundle: SpiderWBundle
    kernels: Any
    config: NativeOrbitConfig = field(default_factory=NativeOrbitConfig)
    row_ptr: Optional[np.ndarray] = None
    col_idx: Optional[np.ndarray] = None
    edge_w: Optional[np.ndarray] = None

    @classmethod
    def from_spiderw(
        cls, path: str, config: Optional[NativeOrbitConfig] = None
    ) -> "NativeBackend":
        if NativeKernels is None:
            raise RuntimeError("spider_native bridge not importable")
        bundle = load_spiderw(path)
        kernels = NativeKernels()
        cfg = config or NativeOrbitConfig(max_loops=bundle.meta.num_loops)
        row_ptr = col_idx = edge_w = None
        if bundle.csr and bundle.csr.get("row_ptr") is not None:
            row_ptr = np.ascontiguousarray(bundle.csr["row_ptr"], dtype=np.int32)
            col_idx = np.ascontiguousarray(bundle.csr["col_idx"], dtype=np.int32)
            edge_w = np.ascontiguousarray(bundle.csr["edge_w"], dtype=np.float32)
        return cls(
            bundle=bundle,
            kernels=kernels,
            config=cfg,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_w=edge_w,
        )

    @property
    def meta(self):
        return self.bundle.meta

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        emb = self.bundle.require("token_emb").astype(np.float32)
        ids = np.ascontiguousarray(token_ids, dtype=np.int64)
        return emb[ids]

    def moe_loop_step(
        self, h: np.ndarray, x0: np.ndarray, loop_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = self.meta
        h = np.ascontiguousarray(h, dtype=np.float32)
        x0 = np.ascontiguousarray(x0, dtype=np.float32)
        B, H = h.shape
        assert H == m.hidden_size

        router_w = self.bundle.require("router").astype(np.float32)
        if router_w.shape[0] != H:
            router_w = router_w.T
        logits = h @ router_w
        loop_bias = self.bundle.require("loop_bias").astype(np.float32)
        bias_row = loop_bias if loop_bias.ndim == 1 else loop_bias[loop_idx]
        route, assign = self.kernels.router(logits, bias_row, k=m.top_k, loop_idx=0)

        if self.row_ptr is None:
            messages = h.copy()
        elif "edge_value" in self.bundle.tensors:
            W_edge = self.bundle.require("edge_value").astype(np.float32)
            if W_edge.ndim == 2:
                W_edge = W_edge.reshape(m.num_experts, H, H)
            if hasattr(self.kernels, "edge_project"):
                projected = self.kernels.edge_project(h, W_edge)
                messages = self.kernels.graph_v2(
                    projected, self.row_ptr, self.col_idx, self.edge_w, assign,
                    E=m.num_experts,
                    loop_idx=min(loop_idx, max(m.num_loops - 1, 0)),
                )
            else:
                projected = _edge_project_np(h, W_edge)
                messages = _graph_v2_np(
                    projected, self.row_ptr, self.col_idx, self.edge_w, assign
                )
        else:
            messages = self.kernels.graph(
                h, self.row_ptr, self.col_idx, self.edge_w, assign, route,
                E=m.num_experts,
                loop_idx=min(loop_idx, max(m.num_loops - 1, 0)),
            )

        W_up = self.bundle.require("expert_up").astype(np.float32)
        W_down = self.bundle.require("expert_down").astype(np.float32)
        delta = self.kernels.expert_ffn(
            messages, assign, route, W_up, W_down, E=m.num_experts
        )

        norm = self.bundle.require("moe_norm").astype(np.float32)
        if norm.ndim == 1:
            gamma, bias = norm, np.zeros_like(norm)
        else:
            gamma, bias = norm[0], norm[1]

        h_next = self.kernels.epilogue(
            h, delta, x0, gamma, bias,
            alpha=self.config.residual_scale,
            beta=self.config.reinject_weight,
            eps=self.config.eps,
        )

        exit_mask = np.zeros((B,), dtype=np.uint8)
        if self.config.early_exit and "exit_head" in self.bundle.tensors:
            proj = self.bundle.require("exit_head").astype(np.float32).reshape(-1)
            exit_mask = self.kernels.early_exit(
                h_next, proj, threshold=self.config.exit_threshold
            )
        return h_next, exit_mask

    def run_loops(self, h: np.ndarray, max_loops: Optional[int] = None) -> np.ndarray:
        x0 = h.copy()
        loops = max_loops or self.config.max_loops
        active = np.ones((h.shape[0],), dtype=bool)
        for loop_idx in range(loops):
            if not active.any():
                break
            h_next, exit_mask = self.moe_loop_step(h, x0, loop_idx)
            h = np.where(active[:, None], h_next, h)
            if self.config.early_exit:
                active = active & (exit_mask == 0)
        return h

    def lm_logits(self, h: np.ndarray) -> np.ndarray:
        w = self.bundle.require("lm_head").astype(np.float32)
        return h @ w.T if w.shape[0] == self.meta.vocab_size else h @ w

    def snapshot(self) -> Dict[str, Any]:
        return {
            "technology": "Orbit-Native",
            "phase": "1.1",
            "backend": getattr(self.kernels, "backend", "unknown"),
            "meta": {
                "hidden": self.meta.hidden_size,
                "experts": self.meta.num_experts,
                "loops": self.meta.num_loops,
                "top_k": self.meta.top_k,
            },
            "has_csr": self.row_ptr is not None,
            "has_edge_value": "edge_value" in self.bundle.tensors,
            "early_exit": self.config.early_exit,
        }


def attach_native_to_orbit_engine(engine: Any, spiderw_path: str, **kwargs) -> Any:
    backend = NativeBackend.from_spiderw(spiderw_path, kwargs.get("config"))
    engine._native_backend = backend  # type: ignore[attr-defined]
    engine.native_snapshot = backend.snapshot  # type: ignore[attr-defined]
    return engine
