"""Forge 0.5 — multi-backend detection + Ascend hooks."""
from __future__ import annotations

import torch

from spider_infer.forge import (
    FORGE_VERSION,
    ForgeConfig,
    ForgeBackend,
    init_forge,
    detect_capabilities,
    pick_backend,
    ascend_available,
)
from spider_infer.forge.backends.ascend import AscendOps


def test_forge_version_0_5():
    assert FORGE_VERSION == "0.5.0"


def test_detect_capabilities_always_has_torch():
    caps = detect_capabilities()
    assert caps.torch is True
    d = caps.to_dict()
    assert "cuda" in d and "triton" in d and "ascend" in d


def test_pick_backend_auto_cpu():
    # On this CI host likely no CUDA — AUTO → TORCH
    b = pick_backend(ForgeBackend.AUTO, device_hint="cpu")
    assert b in (
        ForgeBackend.TORCH,
        ForgeBackend.TRITON,
        ForgeBackend.CUDA,
        ForgeBackend.ASCEND,
    )


def test_pick_backend_respects_torch_request():
    assert pick_backend(ForgeBackend.TORCH) == ForgeBackend.TORCH


def test_init_forge_snapshot_has_capabilities():
    rt = init_forge(ForgeConfig(device="cpu", backend=ForgeBackend.TORCH))
    snap = rt.snapshot()
    assert snap["version"] == "0.5.0"
    assert "capabilities" in snap
    assert snap["capabilities"]["torch"] is True
    assert snap["backend"] == "torch"


def test_ascend_ops_fallback_without_npu():
    ops = AscendOps()
    x = torch.randn(3, 8)
    w = torch.randn(16, 8)
    y = ops.linear(x, w)
    assert y.shape == (3, 16)
    st = ops.stats()
    assert st["available"] == ascend_available()


def test_requested_ascend_degrades_safely():
    # Even if Ascend missing, init must not crash
    rt = init_forge(ForgeConfig(device="npu:0", backend=ForgeBackend.ASCEND))
    assert rt.backend in (ForgeBackend.ASCEND, ForgeBackend.TORCH, ForgeBackend.CUDA, ForgeBackend.TRITON)
    x = torch.randn(2, 4)
    w = torch.randn(8, 4)
    y = rt.gemm.linear(x, w)
    assert y.shape == (2, 8)
