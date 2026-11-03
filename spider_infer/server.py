"""HTTP serving layer for Sentinel (Spider architecture): REST + SSE + web GUI.

Endpoints:
  GET  /                         Web chat UI
  GET  /health
  GET  /ready
  GET  /metrics
  GET  /v1/models
  POST /v1/completions           body.stream=true → OpenAI SSE
  POST /v1/chat/completions      body.stream=true → OpenAI SSE
  POST /agent                    AG-UI 1.0 HTTP+SSE (RunAgentInput → event stream)
  POST /ag-ui                    alias of /agent
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .engine import LLMEngine
try:
    from spider_moe.branding import MODEL_ID as _DEFAULT_MODEL_ID
except Exception:
    _DEFAULT_MODEL_ID = "sentinel"
from .metrics import ServeMetrics
from .sampling_params import SamplingParams
from . import ag_ui

# ---------------------------------------------------------------------------
# Embedded minimal chat UI
# ---------------------------------------------------------------------------

_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sentinel Chat</title>
<style>
  :root {
    --bg: #0f1419; --panel: #1a2332; --border: #2d3a4f;
    --text: #e7ecf3; --muted: #8b9bb4; --accent: #5b9fd4;
    --user: #243044; --bot: #1e2a3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; background: var(--panel);
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header span { color: var(--muted); font-size: 12px; }
  #log {
    flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 720px; padding: 12px 14px; border-radius: 12px; line-height: 1.45;
    white-space: pre-wrap; word-break: break-word; border: 1px solid var(--border);
  }
  .msg.user { align-self: flex-end; background: var(--user); }
  .msg.bot { align-self: flex-start; background: var(--bot); }
  .msg .role { font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; }
  footer {
    border-top: 1px solid var(--border); padding: 12px; background: var(--panel);
    display: flex; gap: 8px; align-items: flex-end;
  }
  textarea {
    flex: 1; resize: none; min-height: 44px; max-height: 160px;
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 12px; font: inherit;
  }
  button {
    background: var(--accent); color: #041018; border: 0; border-radius: 10px;
    padding: 10px 16px; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .row { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
  label { font-size: 12px; color: var(--muted); }
  input[type=number] {
    width: 72px; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px;
  }
</style>
</head>
<body>
<header>
  <h1>Sentinel</h1>
  <span id="status">connecting…</span>
</header>
<div id="log"></div>
<footer>
  <div style="flex:1;display:flex;flex-direction:column;gap:6px">
    <textarea id="input" placeholder="Message…" rows="2"></textarea>
    <div class="row">
      <label>max_tokens <input id="max_tokens" type="number" value="64" min="1" max="512"/></label>
      <label>temp <input id="temp" type="number" value="0.8" min="0" max="2" step="0.1"/></label>
    </div>
  </div>
  <button id="send">Send</button>
</footer>
<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');

async function refreshStatus() {
  try {
    const r = await fetch('/ready');
    const j = await r.json();
    statusEl.textContent = j.ready ? 'ready · streaming on' : 'not ready';
  } catch { statusEl.textContent = 'offline'; }
}
refreshStatus();

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  div.innerHTML = '<div class="role">' + role + '</div>';
  const body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return body;
}

async function streamChat(prompt) {
  const max_tokens = Number(document.getElementById('max_tokens').value) || 64;
  const temperature = Number(document.getElementById('temp').value) || 0.8;
  addMsg('user', prompt);
  const botBody = addMsg('assistant', '');
  sendBtn.disabled = true;
  try {
    const resp = await fetch('/agent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({
        threadId: 'ui-thread',
        runId: 'run-' + Date.now(),
        messages: [{role: 'user', content: prompt}],
        max_tokens, temperature
      })
    });
    if (!resp.ok) {
      const err = await resp.text();
      botBody.textContent = 'Error: ' + err;
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const chunk of parts) {
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;
          try {
            const j = JSON.parse(data);
            if (j.type === 'TEXT_MESSAGE_CONTENT' && j.delta) {
              botBody.textContent += j.delta;
              log.scrollTop = log.scrollHeight;
            } else if (j.type === 'RUN_ERROR') {
              botBody.textContent += '\n[error] ' + (j.message || JSON.stringify(j));
            }
          } catch {}
        }
      }
    }
  } catch (e) {
    botBody.textContent += '\n[stream error] ' + e;
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.onclick = () => {
  const t = input.value.trim();
  if (!t) return;
  input.value = '';
  streamChat(t);
};
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});
</script>
</body>
</html>
"""


class ServeConfig:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_inflight: int = 8,
        default_max_tokens: int = 64,
        hard_max_tokens: int = 512,
        hard_max_prompt_tokens: int = 2048,
        request_timeout_s: float = 120.0,
        model_name: str = _DEFAULT_MODEL_ID,
    ):
        self.host = host
        self.port = port
        self.max_inflight = max_inflight
        self.default_max_tokens = default_max_tokens
        self.hard_max_tokens = hard_max_tokens
        self.hard_max_prompt_tokens = hard_max_prompt_tokens
        self.request_timeout_s = request_timeout_s
        self.model_name = model_name


class InferenceServer:
    """Owns engine + metrics + concurrency gate."""

    def __init__(self, engine: LLMEngine, config: ServeConfig | None = None):
        self.engine = engine
        self.config = config or ServeConfig()
        self.metrics = ServeMetrics()
        self._sem = threading.Semaphore(self.config.max_inflight)
        self._lock = threading.Lock()
        self._ready = True

    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.engine.tokenizer is None:
            ids = [
                b % self.engine.model.cfg.vocab_size for b in prompt.encode("utf-8")
            ] or [1]
        else:
            ids = self.engine.tokenizer.encode(prompt)
        if len(ids) > self.config.hard_max_prompt_tokens:
            raise ValueError(
                f"prompt_tokens={len(ids)} exceeds hard_max_prompt_tokens="
                f"{self.config.hard_max_prompt_tokens}"
            )
        return ids

    def _params(
        self,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
    ) -> SamplingParams:
        mt = max_tokens or self.config.default_max_tokens
        mt = max(1, min(int(mt), self.config.hard_max_tokens))
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=mt,
            seed=seed,
        )

    def _decode_token(self, tid: int) -> str:
        if self.engine.tokenizer is not None:
            return self.engine.tokenizer.decode([tid])
        return str(tid) + " "

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        ids = self._encode_prompt(prompt)
        params = self._params(max_tokens, temperature, top_p, top_k, seed)

        if not self._sem.acquire(blocking=False):
            self.metrics.on_reject()
            raise RuntimeError("server at capacity")
        self.metrics.on_admit()
        t0 = time.perf_counter()
        try:
            with self._lock:
                outs = self.engine.generate([ids], params)
            out = outs[0]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if out.text is not None:
                text = out.text
                # Prefer completion-only text for API parity when tokenizer present
                if self.engine.tokenizer is not None:
                    text = self.engine.tokenizer.decode(out.output_token_ids)
            elif self.engine.tokenizer is not None:
                text = self.engine.tokenizer.decode(out.output_token_ids)
            else:
                text = " ".join(str(t) for t in out.output_token_ids)
            self.metrics.on_success(
                prompt_tokens=len(out.prompt_token_ids),
                gen_tokens=len(out.output_token_ids),
                latency_ms=latency_ms,
            )
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "model": self.config.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop" if out.finished else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(out.prompt_token_ids),
                    "completion_tokens": len(out.output_token_ids),
                    "total_tokens": len(out.token_ids),
                },
                "latency_ms": latency_ms,
            }
        except Exception:
            self.metrics.on_error()
            raise
        finally:
            self._sem.release()

    def stream_tokens(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        seed: int | None = None,
    ) -> Iterator[tuple[str, dict[str, Any] | None]]:
        """Yield (text_piece, final_usage_or_None) as tokens are produced."""
        ids = self._encode_prompt(prompt)
        params = self._params(max_tokens, temperature, top_p, top_k, seed)

        if not self._sem.acquire(blocking=False):
            self.metrics.on_reject()
            raise RuntimeError("server at capacity")
        self.metrics.on_admit()
        t0 = time.perf_counter()
        n_gen = 0
        try:
            # Stream under lock so concurrent requests don't interleave engine state.
            with self._lock:
                for tid in self.engine.generate_stream(ids, params):
                    n_gen += 1
                    yield self._decode_token(tid), None
            latency_ms = (time.perf_counter() - t0) * 1000.0
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": n_gen,
                "total_tokens": len(ids) + n_gen,
                "latency_ms": latency_ms,
            }
            self.metrics.on_success(len(ids), n_gen, latency_ms)
            yield "", usage
        except Exception:
            self.metrics.on_error()
            raise
        finally:
            self._sem.release()


def _sse(data: dict | str) -> bytes:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


def _make_handler(server: InferenceServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, code: int, data: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def _start_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/ui", "/index.html"):
                return self._send_bytes(200, _UI_HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/health":
                return self._send_json(200, {"status": "ok"})
            if path == "/ready":
                code = 200 if server._ready else 503
                return self._send_json(code, {"ready": server._ready})
            if path == "/metrics":
                payload = server.metrics.prometheus()
                try:
                    eng = server.engine.stats()
                    payload += (
                        f'spider_cache_used_pages {eng.get("cache", {}).get("used_pages", 0)}\n'
                        f'spider_cache_free_pages {eng.get("cache", {}).get("free_pages", 0)}\n'
                        f'spider_scheduler_waiting {eng.get("waiting", 0)}\n'
                        f'spider_scheduler_running {eng.get("running", 0)}\n'
                        f'spider_prefix_cache_entries {eng.get("prefix_cache_entries", 0)}\n'
                    )
                except Exception:
                    pass
                return self._send_bytes(
                    200, payload.encode("utf-8"), "text/plain; version=0.0.4"
                )
            if path == "/v1/models":
                return self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server.config.model_name,
                                "object": "model",
                                "owned_by": "sentinel",
                            }
                        ],
                    },
                )
            return self._send_json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._read_json()
            except Exception as exc:
                return self._send_json(400, {"error": f"invalid json: {exc}"})

            stream = bool(body.get("stream", False))
            temperature = float(body.get("temperature", 0.8))
            top_p = float(body.get("top_p", 0.9))
            top_k = int(body.get("top_k", 0) or 0)
            seed = body.get("seed")
            max_tokens = body.get("max_tokens")

            try:
                if path == "/v1/completions":
                    prompt = body.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else ""
                    prompt = str(prompt)
                    if stream:
                        return self._stream_completion(
                            prompt, max_tokens, temperature, top_p, top_k, seed, chat=False
                        )
                    result = server.generate_text(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                    )
                    return self._send_json(200, result)

                if path == "/v1/chat/completions":
                    messages = body.get("messages") or []
                    parts = []
                    for m in messages:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        parts.append(f"{role}: {content}")
                    parts.append("assistant:")
                    prompt = "\n".join(parts)
                    if stream:
                        return self._stream_completion(
                            prompt, max_tokens, temperature, top_p, top_k, seed, chat=True
                        )
                    result = server.generate_text(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                    )
                    text = result["choices"][0]["text"]
                    chat = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "model": server.config.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": result["choices"][0]["finish_reason"],
                            }
                        ],
                        "usage": result["usage"],
                        "latency_ms": result["latency_ms"],
                    }
                    return self._send_json(200, chat)

                if path in ("/agent", "/ag-ui"):
                    return self._ag_ui_run(body)

                return self._send_json(404, {"error": "not found"})
            except RuntimeError as exc:
                if "capacity" in str(exc).lower():
                    return self._send_json(429, {"error": str(exc)})
                return self._send_json(500, {"error": str(exc)})
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                return self._send_json(500, {"error": str(exc)})


        def _ag_ui_run(self, body: dict):
            """AG-UI 1.0: POST RunAgentInput → SSE event stream."""
            try:
                thread_id, run_id, messages, opts = ag_ui.parse_run_input(body)
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})

            prompt = ag_ui.messages_to_prompt(messages)
            self._start_sse()
            try:
                token_stream = server.stream_tokens(
                    prompt,
                    max_tokens=opts.get("max_tokens"),
                    temperature=float(opts.get("temperature") or 0.8),
                    top_p=float(opts.get("top_p") or 0.9),
                    top_k=int(opts.get("top_k") or 0),
                    seed=opts.get("seed"),
                )
                for evt in ag_ui.stream_text_run(
                    token_stream,
                    thread_id=thread_id,
                    run_id=run_id,
                    input_=body,
                ):
                    self.wfile.write(_sse(evt))
                    self.wfile.flush()
            except RuntimeError as exc:
                if "capacity" in str(exc).lower():
                    # stream already started — emit RUN_ERROR
                    self.wfile.write(_sse(ag_ui.run_error(str(exc), thread_id, run_id)))
                    self.wfile.flush()
                else:
                    self.wfile.write(_sse(ag_ui.run_error(str(exc), thread_id, run_id)))
                    self.wfile.flush()
            except Exception as exc:
                traceback.print_exc()
                try:
                    self.wfile.write(_sse(ag_ui.run_error(str(exc), thread_id, run_id)))
                    self.wfile.flush()
                except Exception:
                    pass

        def _stream_completion(
            self, prompt, max_tokens, temperature, top_p, top_k, seed, chat: bool
        ):
            self._start_sse()
            cmpl_id = (
                f"chatcmpl-{uuid.uuid4().hex[:12]}"
                if chat
                else f"cmpl-{uuid.uuid4().hex[:12]}"
            )
            try:
                for piece, usage in server.stream_tokens(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                ):
                    if usage is not None:
                        # final chunk
                        if chat:
                            evt = {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "model": server.config.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    k: usage[k]
                                    for k in (
                                        "prompt_tokens",
                                        "completion_tokens",
                                        "total_tokens",
                                    )
                                },
                            }
                        else:
                            evt = {
                                "id": cmpl_id,
                                "object": "text_completion",
                                "model": server.config.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": "",
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    k: usage[k]
                                    for k in (
                                        "prompt_tokens",
                                        "completion_tokens",
                                        "total_tokens",
                                    )
                                },
                            }
                        self.wfile.write(_sse(evt))
                        self.wfile.write(_sse("[DONE]"))
                        self.wfile.flush()
                        return
                    if not piece:
                        continue
                    if chat:
                        evt = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "model": server.config.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    else:
                        evt = {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "model": server.config.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": piece,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    self.wfile.write(_sse(evt))
                    self.wfile.flush()
            except Exception as exc:
                err = {"error": str(exc)}
                try:
                    self.wfile.write(_sse(err))
                    self.wfile.write(_sse("[DONE]"))
                    self.wfile.flush()
                except Exception:
                    pass

    return Handler


def run_server(engine: LLMEngine, config: ServeConfig | None = None) -> None:
    config = config or ServeConfig()
    app = InferenceServer(engine, config)
    handler = _make_handler(app)
    httpd = ThreadingHTTPServer((config.host, config.port), handler)
    print(
        f"SpiderInfer UI  → http://{config.host}:{config.port}/"
    )
    print(
        f"API             → http://{config.host}:{config.port}/v1/chat/completions"
    )
    print(f"max_inflight={config.max_inflight}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()
