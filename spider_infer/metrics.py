"""Lightweight process metrics for SpiderInfer serving."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ServeMetrics:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_success: int = 0
    requests_error: int = 0
    requests_rejected: int = 0
    tokens_generated: int = 0
    prompt_tokens_total: int = 0
    inflight: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_admit(self) -> None:
        with self._lock:
            self.inflight += 1
            self.requests_total += 1

    def on_success(self, prompt_tokens: int, gen_tokens: int, latency_ms: float) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self.requests_success += 1
            self.prompt_tokens_total += prompt_tokens
            self.tokens_generated += gen_tokens
            self.total_latency_ms += latency_ms

    def on_error(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self.requests_error += 1

    def on_reject(self) -> None:
        with self._lock:
            self.requests_rejected += 1
            self.requests_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = max(1e-6, time.time() - self.started_at)
            avg_lat = (
                self.total_latency_ms / self.requests_success
                if self.requests_success
                else 0.0
            )
            return {
                "uptime_seconds": uptime,
                "requests_total": self.requests_total,
                "requests_success": self.requests_success,
                "requests_error": self.requests_error,
                "requests_rejected": self.requests_rejected,
                "inflight": self.inflight,
                "tokens_generated": self.tokens_generated,
                "prompt_tokens_total": self.prompt_tokens_total,
                "avg_latency_ms": avg_lat,
                "tokens_per_second": self.tokens_generated / uptime,
            }

    def prometheus(self) -> str:
        s = self.snapshot()
        lines = [
            f'spider_uptime_seconds {s["uptime_seconds"]:.3f}',
            f'spider_requests_total {s["requests_total"]}',
            f'spider_requests_success {s["requests_success"]}',
            f'spider_requests_error {s["requests_error"]}',
            f'spider_requests_rejected {s["requests_rejected"]}',
            f'spider_inflight {s["inflight"]}',
            f'spider_tokens_generated {s["tokens_generated"]}',
            f'spider_prompt_tokens_total {s["prompt_tokens_total"]}',
            f'spider_avg_latency_ms {s["avg_latency_ms"]:.3f}',
            f'spider_tokens_per_second {s["tokens_per_second"]:.3f}',
        ]
        return "\n".join(lines) + "\n"
