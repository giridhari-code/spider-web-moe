"""ctypes bridge to libspider_kernels / libspider_cuda.

Does not import torch. Safe for Orbit native path and unit tests.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_CUDA: Optional[ctypes.CDLL] = None


def _candidate_paths(name: str) -> list[Path]:
    roots = [
        Path(__file__).resolve().parents[2],  # repo root if kernels/python/
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
    # bare soname
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


def load_kernels(force: bool = False) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None and not force:
        return _LIB
    for n in (
        "libspider_kernels.so",
        "libspider_kernels.dylib",
        "spider_kernels.dll",
    ):
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
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.spider_router_f32.restype = ctypes.c_int

    lib.spider_graph_message_passing_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_graph_message_passing_f32.restype = ctypes.c_int

    lib.spider_expert_grouped_gemm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_expert_grouped_gemm_f32.restype = ctypes.c_int

    lib.spider_loop_epilogue_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spider_loop_epilogue_f32.restype = ctypes.c_int

    lib.spider_early_exit_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint8),
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
    """High-level NumPy API over the C library."""

    def __init__(self) -> None:
        self.lib = load_kernels()

    @property
    def backend(self) -> str:
        return self.lib.spider_backend_name().decode()

    def router(
        self, logits: np.ndarray, loop_bias: np.ndarray, k: int, loop_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
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
        self,
        hidden: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        edge_w: np.ndarray,
        assign: np.ndarray,
        route: np.ndarray,
        E: int,
        loop_idx: int = 0,
    ) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        B, H = hidden.shape
        K = assign.shape[1]
        nnz = int(col_idx.shape[0])
        out = np.zeros_like(hidden)
        _, hp = _f32_ptr(hidden)
        rp, rpp = _i32_ptr(row_ptr)
        ci, cip = _i32_ptr(col_idx)
        ew, ewp = _f32_ptr(edge_w)
        asg, asgp = _i32_ptr(assign)
        rt, rtp = _f32_ptr(route)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_graph_message_passing_f32(
            hp, B, H, E, K, rpp, cip, ewp, nnz, loop_idx, asgp, rtp, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_graph_message_passing_f32 failed: {rc}")
        return out

    def expert_ffn(
        self,
        tokens: np.ndarray,
        assign: np.ndarray,
        route: np.ndarray,
        W_up: np.ndarray,
        W_down: np.ndarray,
        E: int,
    ) -> np.ndarray:
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        B, H = tokens.shape
        U = W_up.shape[-1]
        K = assign.shape[1]
        out = np.zeros_like(tokens)
        _, tp = _f32_ptr(tokens)
        asg, asgp = _i32_ptr(assign)
        rt, rtp = _f32_ptr(route)
        wu, wup = _f32_ptr(W_up)
        wd, wdp = _f32_ptr(W_down)
        _, op = _f32_ptr(out)
        rc = self.lib.spider_expert_grouped_gemm_f32(
            tp, B, H, U, E, K, asgp, rtp, wup, wdp, op
        )
        if rc != 0:
            raise RuntimeError(f"spider_expert_grouped_gemm_f32 failed: {rc}")
        return out

    def epilogue(
        self,
        h: np.ndarray,
        delta: np.ndarray,
        x0: np.ndarray,
        gamma: np.ndarray,
        bias: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        eps: float = 1e-5,
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
    """edge_indices/weights: [E, M] → CSR row_ptr, col_idx, edge_w."""
    E, M = edge_indices.shape
    row_ptr = [0]
    col_idx = []
    edge_w = []
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
