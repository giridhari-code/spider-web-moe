"""ctypes bridge to libspider_kernels / libspider_cuda (Phase 1–1.3).

Does not import torch. Safe for Orbit native path and unit tests.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_CUDA: Optional[ctypes.CDLL] = None


def _candidate_paths(name: str) -> list[Path]:
    roots = [
        Path(__file__).resolve().parents[2],
        Path.cwd(),
        Path(os.environ.get("SPIDER_KERNEL_LIBDIR", "")),
    ]
    out: list[Path] = []
    for r in roots:
        if not r or str(r) == ".":
            continue
        out.extend(
            [
                r / "build" / name,
                r / "build-cuda" / name,
                r / "build" / "lib" / name,
                r / name,
            ]
        )
    return out


def _load(name: str) -> Optional[ctypes.CDLL]:
    for p in _candidate_paths(name):
        if p.is_file():
            return ctypes.CDLL(str(p))
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


def load_kernels(force: bool = False) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None and not force:
        return _LIB
    for n in ("libspider_kernels.so", "libspider_kernels.dylib", "spider_kernels.dll"):
        lib = _load(n)
        if lib is not None:
            _LIB = lib
            _bind(_LIB)
            return _LIB
    raise RuntimeError(
        "libspider_kernels not found. Build with: "
        "cmake -S . -B build -DSPIDER_BUILD_TESTS=ON && cmake --build build -j"
    )


def load_cuda(force: bool = False) -> Optional[ctypes.CDLL]:
    global _CUDA
    if _CUDA is not None and not force:
        return _CUDA
    for n in ("libspider_cuda.so", "libspider_cuda.dylib", "spider_cuda.dll"):
        lib = _load(n)
        if lib is not None:
            _CUDA = lib
            return _CUDA
    return None


def _bind(lib: ctypes.CDLL) -> None:
    lib.spider_router_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
    ]
    lib.spider_router_f32.restype = ctypes.c_int

    lib.spider_graph_message_passing_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_graph_message_passing_f32.restype = ctypes.c_int

    lib.spider_expert_grouped_gemm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_expert_grouped_gemm_f32.restype = ctypes.c_int

    lib.spider_loop_epilogue_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_loop_epilogue_f32.restype = ctypes.c_int

    lib.spider_early_exit_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.spider_early_exit_f32.restype = ctypes.c_int
    lib.spider_backend_name.restype = ctypes.c_char_p
    if hasattr(lib, "spider_backend_caps"):
        lib.spider_backend_caps.restype = ctypes.c_uint32


def _f32_ptr(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32_ptr(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


class NativeKernels:
    """High-level NumPy API over the C library (Phase 1–1.3)."""

    def __init__(self) -> None:
        self.lib = load_kernels()

    @property
    def backend(self) -> str:
        return self.lib.spider_backend_name().decode()

    def router(
        self, logits: np.ndarray, loop_bias: np.ndarray, k: int, loop_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        bias = np.ascontiguousarray(loop_bias, dtype=np.float32)
        B, E = logits.shape
        w = np.zeros((B, k), dtype=np.float32)
        idx = np.zeros((B, k), dtype=np.int32)
        _, lp = _f32_ptr(logits)
        _, bp = _f32_ptr(bias)
        _, wp = _f32_ptr(w)
        _, ip = _i32_ptr(idx)
        rc = self.lib.spider_router_f32(lp, bp, B, E, k, loop_idx, wp, ip)
        if rc != 0:
            raise RuntimeError(f"spider_router_f32 failed: {rc}")
        return w, idx

    def graph(
        self, hidden, row_ptr, col_idx, edge_w, assign, route, E: int, loop_idx: int = 0
    ) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        B, H = hidden.shape
        K = assign.shape[1]
        nnz = int(col_idx.shape[0])
        out = np.zeros_like(hidden)
        _, hp = _f32_ptr(hidden)
        _, rpp = _i32_ptr(row_ptr)
        _, cip = _i32_ptr(col_idx)
        _, ewp = _f32_ptr(edge_w)
        _, asgp = _i32_ptr(assign)
        _, rtp = _f32_ptr(route)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_graph_message_passing_f32(
            hp, B, H, E, K, rpp, cip, ewp, nnz, loop_idx, asgp, rtp, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_graph_message_passing_f32 failed: {rc}")
        return out

    def edge_project(self, hidden: np.ndarray, W_edge: np.ndarray) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        W_edge = np.ascontiguousarray(W_edge, dtype=np.float32)
        B, H = hidden.shape
        E = W_edge.shape[0]
        out = np.zeros((B, E, H), dtype=np.float32)
        if not hasattr(self.lib, "spider_edge_project_f32"):
            for e in range(E):
                out[:, e, :] = hidden @ W_edge[e].T
            return out
        self.lib.spider_edge_project_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.spider_edge_project_f32.restype = ctypes.c_int
        _, hp = _f32_ptr(hidden)
        _, wp = _f32_ptr(W_edge)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_edge_project_f32(hp, B, H, E, wp, op)
        if rc != 0:
            raise RuntimeError(f"spider_edge_project_f32 failed: {rc}")
        return out

    def graph_v2(
        self, projected, row_ptr, col_idx, edge_w, assign, E: int, loop_idx: int = 0
    ) -> np.ndarray:
        projected = np.ascontiguousarray(projected, dtype=np.float32)
        B, E_, H = projected.shape
        assert E_ == E
        K = assign.shape[1]
        nnz = int(col_idx.shape[0])
        out = np.zeros((B, H), dtype=np.float32)
        if not hasattr(self.lib, "spider_graph_message_passing_v2_f32"):
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
        self.lib.spider_graph_message_passing_v2_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.spider_graph_message_passing_v2_f32.restype = ctypes.c_int
        _, pp = _f32_ptr(projected)
        _, rpp = _i32_ptr(row_ptr)
        _, cip = _i32_ptr(col_idx)
        _, ewp = _f32_ptr(edge_w)
        _, asgp = _i32_ptr(assign)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_graph_message_passing_v2_f32(
            pp, B, H, E, K, rpp, cip, ewp, nnz, loop_idx, asgp, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_graph_message_passing_v2_f32 failed: {rc}")
        return out

    def edge_topk(
        self, edge_logits: np.ndarray, M: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits = np.ascontiguousarray(edge_logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits.reshape(1, *logits.shape)
        B, E, E2 = logits.shape
        assert E == E2
        idx = np.zeros((B, E, M), dtype=np.int32)
        wt = np.zeros((B, E, M), dtype=np.float32)
        if not hasattr(self.lib, "spider_edge_topk_f32"):
            for b in range(B):
                for src in range(E):
                    row = logits[b, src]
                    top = np.argpartition(-row, M - 1)[:M]
                    top = top[np.argsort(-row[top])]
                    idx[b, src] = top
                    x = row[top]
                    x = np.exp(x - x.max())
                    wt[b, src] = x / x.sum()
            return idx, wt
        self.lib.spider_edge_topk_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.spider_edge_topk_f32.restype = ctypes.c_int
        _, lp = _f32_ptr(logits)
        _, ip = _i32_ptr(idx)
        _, wp = _f32_ptr(wt)
        rc = self.lib.spider_edge_topk_f32(lp, B, E, M, ip, wp)
        if rc != 0:
            raise RuntimeError(f"spider_edge_topk_f32 failed: {rc}")
        return idx, wt

    def edge_generator(
        self, hidden: np.ndarray, loop_emb: np.ndarray, W_gen: np.ndarray, E: int
    ) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        loop_emb = np.ascontiguousarray(loop_emb, dtype=np.float32).reshape(-1)
        W_gen = np.ascontiguousarray(W_gen, dtype=np.float32)
        B, H = hidden.shape
        D = int(loop_emb.shape[0])
        out = np.zeros((B, E * E), dtype=np.float32)
        if not hasattr(self.lib, "spider_edge_generator_f32"):
            feat = np.concatenate(
                [hidden, np.broadcast_to(loop_emb, (B, D))], axis=1
            )
            return (feat @ W_gen.T).reshape(B, E, E)
        self.lib.spider_edge_generator_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.spider_edge_generator_f32.restype = ctypes.c_int
        _, hp = _f32_ptr(hidden)
        _, ep = _f32_ptr(loop_emb)
        _, wp = _f32_ptr(W_gen)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_edge_generator_f32(hp, B, H, E, ep, D, wp, op)
        if rc != 0:
            raise RuntimeError(f"spider_edge_generator_f32 failed: {rc}")
        return out.reshape(B, E, E)

    def graph_dynamic(
        self, projected, edge_indices, edge_weights, assign, E: int, M: int
    ) -> np.ndarray:
        projected = np.ascontiguousarray(projected, dtype=np.float32)
        B, E_, H = projected.shape
        assert E_ == E
        K = assign.shape[1]
        out = np.zeros((B, H), dtype=np.float32)
        if not hasattr(self.lib, "spider_graph_message_passing_dynamic_f32"):
            for b in range(B):
                for k in range(K):
                    src = int(assign[b, k])
                    if src < 0:
                        continue
                    for m in range(M):
                        dst = int(edge_indices[b, src, m])
                        out[b] += float(edge_weights[b, src, m]) * projected[b, dst]
            return out
        self.lib.spider_graph_message_passing_dynamic_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.spider_graph_message_passing_dynamic_f32.restype = ctypes.c_int
        _, pp = _f32_ptr(projected)
        _, ip = _i32_ptr(edge_indices)
        _, wp = _f32_ptr(edge_weights)
        _, ap = _i32_ptr(assign)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_graph_message_passing_dynamic_f32(
            pp, B, H, E, K, M, ip, wp, ap, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_graph_message_passing_dynamic_f32 failed: {rc}")
        return out

    def graph_from_generator(
        self,
        hidden: np.ndarray,
        loop_emb: np.ndarray,
        W_gen: np.ndarray,
        W_edge: np.ndarray,
        assign: np.ndarray,
        E: int,
        M: int,
    ) -> np.ndarray:
        logits = self.edge_generator(hidden, loop_emb, W_gen, E)
        edge_idx, edge_w = self.edge_topk(logits, M)
        projected = self.edge_project(hidden, W_edge)
        return self.graph_dynamic(projected, edge_idx, edge_w, assign, E, M)

    def expert_ffn(
        self, tokens, assign, route, W_up, W_down, E: int
    ) -> np.ndarray:
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        B, H = tokens.shape
        U = W_up.shape[-1]
        K = assign.shape[1]
        out = np.zeros_like(tokens)
        _, tp = _f32_ptr(tokens)
        _, asgp = _i32_ptr(assign)
        _, rtp = _f32_ptr(route)
        _, wup = _f32_ptr(W_up)
        _, wdp = _f32_ptr(W_down)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_expert_grouped_gemm_f32(
            tp, B, H, U, E, K, asgp, rtp, wup, wdp, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_expert_grouped_gemm_f32 failed: {rc}")
        return out

    def epilogue(
        self, h, delta, x0, gamma, bias, alpha: float = 1.0, beta: float = 0.0, eps: float = 1e-5
    ) -> np.ndarray:
        h = np.ascontiguousarray(h, dtype=np.float32)
        B, H = h.shape
        out = np.zeros_like(h)
        _, hp = _f32_ptr(h)
        _, dp = _f32_ptr(np.ascontiguousarray(delta, dtype=np.float32))
        _, xp = _f32_ptr(np.ascontiguousarray(x0, dtype=np.float32))
        _, gp = _f32_ptr(np.ascontiguousarray(gamma, dtype=np.float32))
        _, bp = _f32_ptr(np.ascontiguousarray(bias, dtype=np.float32))
        _, op = _f32_ptr(out)
        rc = self.lib.spider_loop_epilogue_f32(
            hp, dp, xp, B, H, float(alpha), float(beta), gp, bp, float(eps), op
        )
        if rc != 0:
            raise RuntimeError(f"spider_loop_epilogue_f32 failed: {rc}")
        return out

    def early_exit(
        self, h: np.ndarray, proj: np.ndarray, threshold: float = 0.85
    ) -> np.ndarray:
        h = np.ascontiguousarray(h, dtype=np.float32)
        B, H = h.shape
        mask = np.zeros((B,), dtype=np.uint8)
        _, hp = _f32_ptr(h)
        _, pp = _f32_ptr(np.ascontiguousarray(proj, dtype=np.float32))
        mp = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        rc = self.lib.spider_early_exit_f32(hp, pp, B, H, float(threshold), mp)
        if rc != 0:
            raise RuntimeError(f"spider_early_exit_f32 failed: {rc}")
        return mask


def build_csr_from_edges(edge_indices: np.ndarray, edge_weights: np.ndarray):
    E, M = edge_indices.shape
    row_ptr = [0]
    col_idx, edge_w = [], []
    for src in range(E):
        for m in range(M):
            col_idx.append(int(edge_indices[src, m]))
            edge_w.append(float(edge_weights[src, m]))
        row_ptr.append(len(col_idx))
    return (
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(col_idx, dtype=np.int32),
        np.asarray(edge_w, dtype=np.float32),
    )


if __name__ == "__main__":
    try:
        nk = NativeKernels()
        print("backend:", nk.backend)
        logits = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        w, idx = nk.router(logits, bias, k=2)
        print("router weights", w, "idx", idx)
    except Exception as e:
        print("native load failed:", e, file=sys.stderr)
        sys.exit(1)
