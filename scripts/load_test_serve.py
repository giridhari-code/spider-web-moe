#!/usr/bin/env python3
"""Simple concurrent load test against SpiderInfer HTTP server."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request


def one_request(url: str, prompt: str, max_tokens: int, results: list, lock: threading.Lock):
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        url + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            lat = (time.perf_counter() - t0) * 1000
            with lock:
                results.append({"ok": True, "latency_ms": lat, "usage": data.get("usage")})
    except Exception as exc:
        with lock:
            results.append({"ok": False, "error": str(exc)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--prompt", default="The history of science")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    # health
    with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
        print("health", r.read().decode())

    results: list = []
    lock = threading.Lock()
    threads = []
    t0 = time.perf_counter()
    for i in range(args.requests):
        while threading.active_count() > args.concurrency + 1:
            time.sleep(0.01)
        th = threading.Thread(
            target=one_request,
            args=(args.url, args.prompt, args.max_tokens, results, lock),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t0
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    lats = sorted(r["latency_ms"] for r in ok)
    def pct(p):
        if not lats:
            return None
        return lats[min(len(lats) - 1, int(p * (len(lats) - 1)))]
    print(json.dumps({
        "elapsed_s": elapsed,
        "ok": len(ok),
        "errors": len(bad),
        "rps": len(results) / max(elapsed, 1e-6),
        "latency_ms_p50": pct(0.50),
        "latency_ms_p95": pct(0.95),
        "sample_error": bad[0]["error"] if bad else None,
    }, indent=2))


if __name__ == "__main__":
    main()
