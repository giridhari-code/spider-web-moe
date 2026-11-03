"""Exact-prompt prefix cache using GlobalPagedKVCache.fork_prefix.

When a new request's full prompt matches a previously completed (or still
resident) request's prompt, we share physical KV pages via refcount + COW
instead of recomputing the prefill.

Eviction is LRU under *two* ceilings:
  - max_entries (cardinality)
  - max_bytes   (estimated resident KV for cached prefixes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable


@dataclass
class PrefixEntry:
    source_request_id: Hashable
    prompt_tokens: tuple[int, ...]
    length: int
    approx_bytes: int = 0


class PrefixCache:
    """In-memory exact-match prompt → resident request mapping."""

    def __init__(
        self,
        max_entries: int = 128,
        max_bytes: int | None = None,
        bytes_per_token: int = 0,
        on_evict: Callable[[Hashable], None] | None = None,
    ):
        """
        Parameters
        ----------
        max_entries:
            Soft cap on number of cached prompts.
        max_bytes:
            Soft cap on sum of approx_bytes across entries. ``None`` = unlimited.
        bytes_per_token:
            Estimated KV bytes per cached prompt token (shell layers × K+V).
            Used when insert() does not pass an explicit approx_bytes.
        on_evict:
            Optional callback(source_request_id) when an entry is dropped so
            the engine can free the holder KV pages.
        """
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max_bytes
        self.bytes_per_token = max(0, int(bytes_per_token))
        self.on_evict = on_evict
        self._map: dict[tuple[int, ...], PrefixEntry] = {}
        self._order: list[tuple[int, ...]] = []  # LRU (oldest at index 0)
        self._total_bytes = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._map)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def lookup(self, prompt_tokens: list[int]) -> PrefixEntry | None:
        key = tuple(prompt_tokens)
        entry = self._map.get(key)
        if entry is None:
            return None
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        return entry

    def _estimate_bytes(self, length: int, approx_bytes: int | None) -> int:
        if approx_bytes is not None and approx_bytes > 0:
            return int(approx_bytes)
        if self.bytes_per_token > 0:
            return int(length * self.bytes_per_token)
        # Fallback: token count as a weak proxy (keeps byte path exercised)
        return int(length)

    def _drop_key(self, key: tuple[int, ...]) -> None:
        entry = self._map.pop(key, None)
        if entry is None:
            return
        self._total_bytes = max(0, self._total_bytes - entry.approx_bytes)
        if key in self._order:
            self._order.remove(key)
        self.evictions += 1
        if self.on_evict is not None:
            try:
                self.on_evict(entry.source_request_id)
            except Exception:
                pass

    def _evict_while_needed(
        self, incoming_bytes: int = 0, *, make_room_for_new: bool = False
    ) -> None:
        # Cardinality: if inserting a new key, free one slot when already full.
        if make_room_for_new:
            while len(self._map) >= self.max_entries and self._order:
                self._drop_key(self._order[0])
        else:
            while len(self._map) > self.max_entries and self._order:
                self._drop_key(self._order[0])
        # Byte budget (leave room for the incoming entry when make_room_for_new)
        if self.max_bytes is not None:
            while self._order and self._total_bytes + incoming_bytes > self.max_bytes:
                if len(self._map) <= 1 and not make_room_for_new:
                    break  # keep at least the current sole entry
                self._drop_key(self._order[0])

    def insert(
        self,
        request_id: Hashable,
        prompt_tokens: list[int],
        length: int,
        approx_bytes: int | None = None,
    ) -> None:
        key = tuple(prompt_tokens)
        nbytes = self._estimate_bytes(length, approx_bytes)

        if key in self._map:
            old = self._map[key]
            self._total_bytes = max(0, self._total_bytes - old.approx_bytes)
            if key in self._order:
                self._order.remove(key)
        else:
            self._evict_while_needed(incoming_bytes=nbytes, make_room_for_new=True)

        entry = PrefixEntry(
            source_request_id=request_id,
            prompt_tokens=key,
            length=length,
            approx_bytes=nbytes,
        )
        self._map[key] = entry
        self._order.append(key)
        self._total_bytes += nbytes
        # Enforce budgets after insert (do not strip the only remaining entry for bytes)
        self._evict_while_needed(incoming_bytes=0, make_room_for_new=False)

    def evict_request(self, request_id: Hashable) -> None:
        dead = [k for k, v in self._map.items() if v.source_request_id == request_id]
        for k in dead:
            # Don't double-call on_evict for the same live free path unless wanted;
            # engine already frees finished seqs. Only remove map entries.
            entry = self._map.pop(k, None)
            if entry is None:
                continue
            self._total_bytes = max(0, self._total_bytes - entry.approx_bytes)
            if k in self._order:
                self._order.remove(k)

    def stats(self) -> dict:
        return {
            "entries": len(self._map),
            "max_entries": self.max_entries,
            "total_bytes": self._total_bytes,
            "max_bytes": self.max_bytes,
            "bytes_per_token": self.bytes_per_token,
            "evictions": self.evictions,
        }
