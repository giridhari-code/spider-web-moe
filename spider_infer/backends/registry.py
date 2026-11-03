from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_backend(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known={list(_REGISTRY)}")
    return _REGISTRY[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)
