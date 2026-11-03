"""AG-UI (Agent-User Interaction Protocol) 1.0 — minimal producer.

Implements the HTTP+SSE binding and the mandatory run lifecycle plus
text-message streaming family:

  RUN_STARTED → TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT* → TEXT_MESSAGE_END → RUN_FINISHED
  (or RUN_ERROR)

Spec reference: https://docs.ag-ui.com/
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any


PROTOCOL_VERSION = "1.0"


def _ts() -> int:
    return int(time.time() * 1000)


def event(type_: str, **fields: Any) -> dict[str, Any]:
    """Build a BaseEvent envelope."""
    payload: dict[str, Any] = {"type": type_, "timestamp": _ts()}
    payload.update(fields)
    return payload


def run_started(thread_id: str, run_id: str, input_: dict | None = None) -> dict:
    e = event("RUN_STARTED", threadId=thread_id, runId=run_id)
    if input_ is not None:
        e["input"] = input_
    return e


def run_finished(thread_id: str, run_id: str, outcome: dict | None = None) -> dict:
    e = event("RUN_FINISHED", threadId=thread_id, runId=run_id)
    e["outcome"] = outcome or {"type": "success"}
    return e


def run_error(message: str, thread_id: str | None = None, run_id: str | None = None) -> dict:
    e = event("RUN_ERROR", message=message)
    if thread_id is not None:
        e["threadId"] = thread_id
    if run_id is not None:
        e["runId"] = run_id
    return e


def text_message_start(message_id: str, role: str = "assistant") -> dict:
    return event("TEXT_MESSAGE_START", messageId=message_id, role=role)


def text_message_content(message_id: str, delta: str) -> dict:
    return event("TEXT_MESSAGE_CONTENT", messageId=message_id, delta=delta)


def text_message_end(message_id: str) -> dict:
    return event("TEXT_MESSAGE_END", messageId=message_id)


def messages_snapshot(messages: list[dict]) -> dict:
    return event("MESSAGES_SNAPSHOT", messages=messages)


def parse_run_input(body: dict) -> tuple[str, str, list[dict], dict]:
    """Extract threadId, runId, messages, and forwarded options from RunAgentInput."""
    thread_id = str(body.get("threadId") or f"thread-{uuid.uuid4().hex[:10]}")
    run_id = str(body.get("runId") or f"run-{uuid.uuid4().hex[:10]}")
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    # Optional generation knobs (AG-UI allows application fields via metadata / custom)
    opts = {
        "max_tokens": body.get("maxTokens") or body.get("max_tokens"),
        "temperature": body.get("temperature", 0.8),
        "top_p": body.get("top_p", body.get("topP", 0.9)),
        "top_k": body.get("top_k", body.get("topK", 0)),
        "seed": body.get("seed"),
    }
    return thread_id, run_id, messages, opts


def messages_to_prompt(messages: list[dict]) -> str:
    """Fold AG-UI / chat messages into a single prompt string."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # content parts array — join text pieces
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    texts.append(part)
            content = "".join(texts)
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def stream_text_run(
    token_stream: Iterator[tuple[str, dict | None]],
    *,
    thread_id: str,
    run_id: str,
    input_: dict | None = None,
) -> Iterator[dict]:
    """Map a (piece, usage|None) token stream into AG-UI events."""
    yield run_started(thread_id, run_id, input_=input_)
    message_id = f"msg-{uuid.uuid4().hex[:12]}"
    yield text_message_start(message_id, role="assistant")
    try:
        for piece, usage in token_stream:
            if usage is not None:
                break
            if piece:
                yield text_message_content(message_id, piece)
        yield text_message_end(message_id)
        yield run_finished(thread_id, run_id, outcome={"type": "success"})
    except Exception as exc:
        yield run_error(str(exc), thread_id=thread_id, run_id=run_id)
