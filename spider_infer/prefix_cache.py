"""Exact-prompt prefix cache using GlobalPagedKVCache.fork_prefix.

When a new request's full prompt matches a previously completed (or still
resident) request's prompt, we share physical KV pages via refcount + COW
instead of recomputing the prefill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


@dataclass
class PrefixEntry:
    source_request_id: Hashable
    prompt_tokens: tuple[int, ...]
    length: int


class PrefixCache:
    """In-memory exact-match prompt → resident request mapping."""

    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._map: dict[tuple[int, ...], PrefixEntry] = {}
        self._order: list[tuple[int, ...]] = []  # LRU

    def __len__(self) -> int:
        return len(self._map)

    def lookup(self, prompt_tokens: list[int]) -> PrefixEntry | None:
        key = tuple(prompt_tokens)
        entry = self._map.get(key)
        if entry is None:
            return None
        # Refresh LRU
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        return entry

    def insert(self, request_id: Hashable, prompt_tokens: list[int], length: int) -> None:
        key = tuple(prompt_tokens)
        if key in self._map:
            if key in self._order:
                self._order.remove(key)
        elif len(self._map) >= self.max_entries and self._order:
            old = self._order.pop(0)
            self._map.pop(old, None)
        self._map[key] = PrefixEntry(
            source_request_id=request_id,
            prompt_tokens=key,
            length=length,
        )
        self._order.append(key)

    def evict_request(self, request_id: Hashable) -> None:
        dead = [k for k, v in self._map.items() if v.source_request_id == request_id]
        for k in dead:
            self._map.pop(k, None)
            if k in self._order:
                self._order.remove(k)
