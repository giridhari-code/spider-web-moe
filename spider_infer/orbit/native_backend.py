"""Orbit native compute backend — Phase 1–2.2.

MoE loops via C kernels; optional causal attention + KV for prelude/coda shells.
`generate()` is greedy decode without torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    B, H = hidden.shape
    E = W_edge.shape[0]
    out = np.zeros((B, E, H), dtype=np.float32)
    for e in range(E):
        out[:, e, :] = hidden @ W_edge[e].T
    return out


def _graph_v2_np(projected, row_ptr, col_idx, edge_w, assign) -> np.ndarray:
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


def _attention_causal_np(
    x: np.ndarray, W_qkv: np.ndarray, W_o: np.ndarray, n_heads: int
) -> np.ndarray:
    """x [B,T,H] full causal (no cache)."""
    B, T, H = x.shape
    D = H // n_heads
    scale = 1.0 / np.sqrt(D)
    qkv = x @ W_qkv.T  # [B,T,3H]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = np.triu(np.ones((T, T), dtype=np.bool_), k=1)
    scores = np.where(mask, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-20)
    ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, H)
    return ctx @ W_o.T


def _attention_kv_np(
    x: np.ndarray,
    W_qkv: np.ndarray,
    W_o: np.ndarray,
    n_heads: int,
    keys: np.ndarray,
    values: np.ndarray,
    position: int,
) -> Tuple[np.ndarray, int]:
    """x [B,T,H]; keys/values [B,n_heads,max_T,D]. Returns out [B,T,H], new_length."""
    B, T, H = x.shape
    D = H // n_heads
    max_T = keys.shape[2]
    if position + T > max_T:
        raise ValueError("KV capacity exceeded")
    scale = 1.0 / np.sqrt(D)
    qkv = x @ W_qkv.T
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)  # B,h,T,D
    k = k.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)
    keys[:, :, position : position + T, :] = k
    values[:, :, position : position + T, :] = v
    new_len = position + T
    out = np.zeros((B, T, H), dtype=np.float32)
    for ti in range(T):
        abs_q = position + ti
        # q[:, :, ti] vs keys[:, :, :abs_q+1]
        qi = q[:, :, ti : ti + 1, :]  # B,h,1,D
        kk = keys[:, :, : abs_q + 1, :]  # B,h,L,D
        vv = values[:, :, : abs_q + 1, :]
        scores = (qi @ kk.transpose(0, 1, 3, 2)) * scale  # B,h,1,L
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-20)
        ctx = (attn @ vv).reshape(B, n_heads, D)
        out[:, ti, :] = ctx.transpose(0, 1, 2).reshape(B, H)
    return out @ W_o.T, new_len


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


@dataclass
class LayerKV:
    keys: np.ndarray
    values: np.ndarray
    length: int = 0

    def reset(self) -> None:
        self.length = 0


@dataclass
class NativeOrbitConfig:
    max_loops: int = 4
    early_exit: bool = True
    exit_threshold: float = 0.85
    residual_scale: float = 1.0
    reinject_weight: float = 0.0
    eps: float = 1e-5
    prefer_cuda: bool = True
    max_seq_len: int = 2048


@dataclass
class NativeBackend:
    bundle: SpiderWBundle
    kernels: Any
    config: NativeOrbitConfig = field(default_factory=NativeOrbitConfig)
    row_ptr: Optional[np.ndarray] = None
    col_idx: Optional[np.ndarray] = None
    edge_w: Optional[np.ndarray] = None
    _prelude_kv: List[LayerKV] = field(default_factory=list)
    _coda_kv: List[LayerKV] = field(default_factory=list)

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

    def _n_heads(self) -> int:
        return max(int(self.meta.num_heads), 1)

    def _ensure_kv(self, n_prelude: int, n_coda: int, B: int) -> None:
        H = self.meta.hidden_size
        nh = self._n_heads()
        D = H // nh
        max_T = self.config.max_seq_len
        if len(self._prelude_kv) != n_prelude or (
            self._prelude_kv and self._prelude_kv[0].keys.shape[0] != B
        ):
            self._prelude_kv = [
                LayerKV(
                    np.zeros((B, nh, max_T, D), dtype=np.float32),
                    np.zeros((B, nh, max_T, D), dtype=np.float32),
                )
                for _ in range(n_prelude)
            ]
        if len(self._coda_kv) != n_coda or (
            self._coda_kv and self._coda_kv[0].keys.shape[0] != B
        ):
            self._coda_kv = [
                LayerKV(
                    np.zeros((B, nh, max_T, D), dtype=np.float32),
                    np.zeros((B, nh, max_T, D), dtype=np.float32),
                )
                for _ in range(n_coda)
            ]

    def reset_kv(self) -> None:
        for c in self._prelude_kv + self._coda_kv:
            c.reset()

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        emb = self.bundle.require("token_emb").astype(np.float32)
        ids = np.ascontiguousarray(token_ids, dtype=np.int64)
        return emb[ids]

    def _attn_block(
        self,
        x: np.ndarray,
        prefix: str,
        layer_i: int,
        kv: LayerKV,
        position: int,
    ) -> np.ndarray:
        """x [B,T,H] residual attn if weights exist else identity."""
        key_qkv = f"{prefix}.{layer_i}.W_qkv"
        key_o = f"{prefix}.{layer_i}.W_o"
        if key_qkv not in self.bundle.tensors or key_o not in self.bundle.tensors:
            return x
        W_qkv = self.bundle.require(key_qkv).astype(np.float32)
        W_o = self.bundle.require(key_o).astype(np.float32)
        nh = self._n_heads()
        # optional pre-LN
        ln_w = self.bundle.tensors.get(f"{prefix}.{layer_i}.ln1_w")
        ln_b = self.bundle.tensors.get(f"{prefix}.{layer_i}.ln1_b")
        h = x
        if ln_w is not None:
            h = _layer_norm(
                x,
                ln_w.astype(np.float32),
                (ln_b if ln_b is not None else np.zeros_like(ln_w)).astype(np.float32),
                self.config.eps,
            )
        y, new_len = _attention_kv_np(
            h, W_qkv, W_o, nh, kv.keys, kv.values, position
        )
        kv.length = new_len
        x = x + y
        # optional FFN
        up = self.bundle.tensors.get(f"{prefix}.{layer_i}.W_ffn_up")
        down = self.bundle.tensors.get(f"{prefix}.{layer_i}.W_ffn_down")
        if up is not None and down is not None:
            ln2_w = self.bundle.tensors.get(f"{prefix}.{layer_i}.ln2_w")
            ln2_b = self.bundle.tensors.get(f"{prefix}.{layer_i}.ln2_b")
            h2 = x
            if ln2_w is not None:
                h2 = _layer_norm(
                    x,
                    ln2_w.astype(np.float32),
                    (ln2_b if ln2_b is not None else np.zeros_like(ln2_w)).astype(
                        np.float32
                    ),
                    self.config.eps,
                )
            x = x + (_gelu(h2 @ up.T) @ down.T)
        return x.astype(np.float32)

    def moe_loop_step(
        self, h: np.ndarray, x0: np.ndarray, loop_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = self.meta
        h = np.ascontiguousarray(h, dtype=np.float32)
        x0 = np.ascontiguousarray(x0, dtype=np.float32)
        B, H = h.shape

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
                    projected,
                    self.row_ptr,
                    self.col_idx,
                    self.edge_w,
                    assign,
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
                h,
                self.row_ptr,
                self.col_idx,
                self.edge_w,
                assign,
                route,
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
            h,
            delta,
            x0,
            gamma,
            bias,
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
        """h [B,H] or [B,T,H] — MoE on last token positions flattened."""
        if h.ndim == 3:
            B, T, H = h.shape
            flat = h.reshape(B * T, H)
            out = self._run_loops_flat(flat, max_loops)
            return out.reshape(B, T, H)
        return self._run_loops_flat(h, max_loops)

    def _run_loops_flat(
        self, h: np.ndarray, max_loops: Optional[int] = None
    ) -> np.ndarray:
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

    def forward_sequence(self, token_ids: np.ndarray) -> np.ndarray:
        """Prefill: token_ids [B,T] → hidden [B,T,H]."""
        ids = np.atleast_2d(np.asarray(token_ids, dtype=np.int64))
        B, T = ids.shape
        n_pre = int(getattr(self.meta, "num_prelude_layers", 0) or 0)
        n_coda = int(getattr(self.meta, "num_coda_layers", 0) or 0)
        self._ensure_kv(n_pre, n_coda, B)
        self.reset_kv()

        x = self.embed_tokens(ids)  # [B,T,H]
        pos = 0
        for i in range(n_pre):
            x = self._attn_block(x, "prelude", i, self._prelude_kv[i], pos)
        # MoE on each position (batched flat)
        x = self.run_loops(x)
        for i in range(n_coda):
            x = self._attn_block(x, "coda", i, self._coda_kv[i], pos)
        # mark KV lengths after prefill
        for kv in self._prelude_kv + self._coda_kv:
            if kv.length == 0:
                kv.length = T
        return x

    def forward_decode_step(self, token_ids: np.ndarray, position: int) -> np.ndarray:
        """Decode one step: token_ids [B] or [B,1] → hidden [B,H]."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        B = ids.shape[0]
        n_pre = int(getattr(self.meta, "num_prelude_layers", 0) or 0)
        n_coda = int(getattr(self.meta, "num_coda_layers", 0) or 0)
        self._ensure_kv(n_pre, n_coda, B)
        x = self.embed_tokens(ids).reshape(B, 1, -1)
        for i in range(n_pre):
            x = self._attn_block(x, "prelude", i, self._prelude_kv[i], position)
        x = self.run_loops(x)
        for i in range(n_coda):
            x = self._attn_block(x, "coda", i, self._coda_kv[i], position)
        return x[:, 0, :]

    def lm_logits(self, h: np.ndarray) -> np.ndarray:
        if "lm_head" in self.bundle.tensors:
            w = self.bundle.require("lm_head").astype(np.float32)
        else:
            w = self.bundle.require("token_emb").astype(np.float32)
        if h.ndim == 3:
            return h @ w.T if w.shape[0] == self.meta.vocab_size else h @ w
        return h @ w.T if w.shape[0] == self.meta.vocab_size else h @ w

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 32,
        eos_id: Optional[int] = None,
    ) -> np.ndarray:
        """Greedy generate. prompt_ids [T] or [B,T] → full sequence ids [B, T+new]."""
        ids = np.atleast_2d(np.asarray(prompt_ids, dtype=np.int64))
        B, T0 = ids.shape
        h = self.forward_sequence(ids)
        logits = self.lm_logits(h[:, -1, :])
        out = [ids[b].tolist() for b in range(B)]
        for step in range(max_new_tokens):
            next_id = np.argmax(logits, axis=-1).astype(np.int64)
            for b in range(B):
                out[b].append(int(next_id[b]))
            if eos_id is not None and np.all(next_id == eos_id):
                break
            pos = T0 + step
            h_step = self.forward_decode_step(next_id, position=pos)
            logits = self.lm_logits(h_step)
        max_len = max(len(o) for o in out)
        arr = np.zeros((B, max_len), dtype=np.int64)
        for b, seq in enumerate(out):
            arr[b, : len(seq)] = seq
        return arr if B > 1 else arr[0]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "technology": "Orbit-Native",
            "phase": "2.2",
            "backend": getattr(self.kernels, "backend", "unknown"),
            "meta": {
                "hidden": self.meta.hidden_size,
                "experts": self.meta.num_experts,
                "loops": self.meta.num_loops,
                "top_k": self.meta.top_k,
                "prelude": int(getattr(self.meta, "num_prelude_layers", 0) or 0),
                "coda": int(getattr(self.meta, "num_coda_layers", 0) or 0),
            },
            "has_csr": self.row_ptr is not None,
            "has_edge_value": "edge_value" in self.bundle.tensors,
            "early_exit": self.config.early_exit,
            "max_seq_len": self.config.max_seq_len,
        }


def attach_native_to_orbit_engine(engine: Any, spiderw_path: str, **kwargs) -> Any:
    backend = NativeBackend.from_spiderw(spiderw_path, kwargs.get("config"))
    engine._native_backend = backend  # type: ignore[attr-defined]
    engine.native_snapshot = backend.snapshot  # type: ignore[attr-defined]
    engine.native_generate = backend.generate  # type: ignore[attr-defined]
    return engine
