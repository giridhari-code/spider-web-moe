"""Request / sequence state for continuous batching."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import itertools
import time

from .sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"


_id_counter = itertools.count()


def _next_request_id() -> str:
    return f"req-{next(_id_counter)}"


@dataclass
class Sequence:
    """One generation request tracked by the engine."""

    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    request_id: str = field(default_factory=_next_request_id)
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # How many tokens have been processed by the model (prompt + generated).
    num_computed_tokens: int = 0

    # Arrival time for simple FCFS / aging policies.
    arrival_time: float = field(default_factory=time.time)

    # Optional user metadata.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def needs_prefill(self) -> bool:
        """True until the full prompt has been computed at least once."""
        return self.num_computed_tokens < self.num_prompt_tokens

    def get_next_token_ids(self) -> list[int]:
        """Tokens that still need to be fed to the model."""
        if self.needs_prefill:
            return self.prompt_token_ids[self.num_computed_tokens :]
        if self.output_token_ids:
            return [self.output_token_ids[-1]]
        return []

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self, eos_token_id: int | None) -> bool:
        """Return True if generation should stop for this sequence."""
        if self.num_output_tokens >= self.sampling_params.max_tokens:
            return True
        if not self.output_token_ids:
            return False
        last = self.output_token_ids[-1]
        if last in self.sampling_params.stop_token_ids:
            return True
        if (
            eos_token_id is not None
            and not self.sampling_params.ignore_eos
            and last == eos_token_id
        ):
            return True
        return False

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED

    def mark_running(self) -> None:
        self.status = SequenceStatus.RUNNING

    def mark_waiting(self) -> None:
        self.status = SequenceStatus.WAITING

    def mark_preempted(self) -> None:
        self.status = SequenceStatus.PREEMPTED
