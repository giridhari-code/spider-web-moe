"""Phase-3 continuous batching scheduler.

Adds:
- Chunked prefill (bounded tokens per step per sequence)
- Equal-length batching for decode / extend
- Preemption of in-progress prefills when decode needs memory
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..config import SchedulerConfig
from ..memory.block_manager import AllocStatus, BlockManager
from ..sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """What the engine should run on this step."""

    scheduled: list[Sequence] = field(default_factory=list)
    # True → prefill or extend (prompt tokens). False → decode (1 new token).
    is_prefill: bool = False
    num_batched_tokens: int = 0
    finished: list[Sequence] = field(default_factory=list)
    # Sequences whose KV was freed and must restart prefill.
    preempted: list[Sequence] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        config: SchedulerConfig,
        max_model_len: int,
    ):
        self.block_manager = block_manager
        self.config = config
        self.max_model_len = max_model_len

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self._preempt_counts: dict[str, int] = {}

    def add_request(self, seq: Sequence) -> None:
        if seq.num_prompt_tokens > self.max_model_len:
            raise ValueError(
                f"request {seq.request_id} prompt length {seq.num_prompt_tokens} "
                f"exceeds max_model_len={self.max_model_len}"
            )
        seq.mark_waiting()
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        finished: list[Sequence] = []
        preempted: list[Sequence] = []

        decode_candidates = [
            s
            for s in self.running
            if s.status == SequenceStatus.RUNNING and not s.needs_prefill
        ]

        if decode_candidates and self.config.prefer_decode:
            out = self._schedule_decode(decode_candidates, finished, preempted)
            if out.scheduled:
                return out

        prefill_candidates = [s for s in self.running if s.needs_prefill] + list(
            self.waiting
        )
        if prefill_candidates:
            out = self._schedule_prefill(prefill_candidates, finished)
            if out.scheduled:
                return out

        if decode_candidates:
            return self._schedule_decode(decode_candidates, finished, preempted)

        return SchedulerOutput(finished=finished, preempted=preempted)

    def free_finished(self, sequences: list[Sequence]) -> None:
        for seq in sequences:
            self.block_manager.free(seq.request_id)
            self._preempt_counts.pop(seq.request_id, None)
            if seq in self.running:
                self.running.remove(seq)
            if seq in self.waiting:
                self.waiting.remove(seq)

    def _schedule_decode(
        self,
        candidates: list[Sequence],
        finished: list[Sequence],
        preempted: list[Sequence],
    ) -> SchedulerOutput:
        by_length: dict[int, list[Sequence]] = defaultdict(list)
        for seq in candidates:
            try:
                length = self.block_manager.sequence_length(seq.request_id)
            except KeyError:
                continue
            by_length[length].append(seq)

        if not by_length:
            return SchedulerOutput(finished=finished, preempted=preempted)

        best_length = max(by_length, key=lambda L: len(by_length[L]))
        group = by_length[best_length]

        scheduled: list[Sequence] = []
        budget = self.config.max_num_batched_tokens

        for seq in group:
            if len(scheduled) >= self.config.max_num_seqs or budget < 1:
                break
            status = self.block_manager.can_append(seq.request_id, extra_tokens=1)
            if status is AllocStatus.NEVER:
                seq.mark_finished()
                finished.append(seq)
                continue
            if status is AllocStatus.LATER:
                if self.config.enable_preemption:
                    if self._try_preempt_for_decode(seq, preempted):
                        status = self.block_manager.can_append(
                            seq.request_id, extra_tokens=1
                        )
                if status is not AllocStatus.OK:
                    continue
            scheduled.append(seq)
            budget -= 1

        return SchedulerOutput(
            scheduled=scheduled,
            is_prefill=False,
            num_batched_tokens=len(scheduled),
            finished=finished,
            preempted=preempted,
        )

    def _try_preempt_for_decode(
        self, needy: Sequence, preempted: list[Sequence]
    ) -> bool:
        victims = [
            s
            for s in self.running
            if s is not needy
            and s.needs_prefill
            and s.num_output_tokens == 0
            and self.block_manager.has_request(s.request_id)
        ]
        victims.sort(key=lambda s: s.num_computed_tokens)

        for victim in victims:
            count = self._preempt_counts.get(victim.request_id, 0)
            if count >= self.config.max_preemptions:
                victim.mark_finished()
                self.block_manager.free(victim.request_id)
                if victim in self.running:
                    self.running.remove(victim)
                continue

            self.block_manager.free(victim.request_id)
            victim.num_computed_tokens = 0
            victim.mark_preempted()
            self._preempt_counts[victim.request_id] = count + 1
            if victim in self.running:
                self.running.remove(victim)
            victim.mark_waiting()
            self.waiting.append(victim)
            preempted.append(victim)
            return True
        return False

    def _chunk_size(self, remaining: int) -> int:
        cs = self.config.prefill_chunk_size
        if cs <= 0:
            return remaining
        return min(remaining, cs)

    def _schedule_prefill(
        self, candidates: list[Sequence], finished: list[Sequence]
    ) -> SchedulerOutput:
        annotated: list[tuple[int, Sequence, int]] = []
        for seq in candidates:
            remaining = seq.num_prompt_tokens - seq.num_computed_tokens
            if remaining <= 0:
                continue
            if self.block_manager.has_request(seq.request_id):
                length = self.block_manager.sequence_length(seq.request_id)
            else:
                length = 0
            annotated.append((length, seq, remaining))

        if not annotated:
            return SchedulerOutput(finished=finished)

        by_len: dict[int, list[tuple[Sequence, int]]] = defaultdict(list)
        for length, seq, remaining in annotated:
            by_len[length].append((seq, remaining))

        def group_key(L: int) -> tuple[int, int]:
            return (len(by_len[L]), 1 if L == 0 else 0)

        best_len = max(by_len.keys(), key=group_key)
        group = by_len[best_len]

        scheduled: list[Sequence] = []
        budget = self.config.max_num_batched_tokens
        watermark = self.config.prefill_watermark
        tokens_this_step = 0

        for seq, remaining in group:
            if len(scheduled) >= self.config.max_num_seqs:
                break
            chunk = self._chunk_size(remaining)
            if chunk > budget and scheduled:
                break

            if not self.block_manager.has_request(seq.request_id):
                needed = min(
                    seq.num_prompt_tokens + seq.sampling_params.max_tokens,
                    self.max_model_len,
                )
                status = self.block_manager.can_allocate(needed)
                if status is AllocStatus.NEVER:
                    seq.mark_finished()
                    finished.append(seq)
                    if seq in self.waiting:
                        self.waiting.remove(seq)
                    continue
                if status is AllocStatus.LATER:
                    continue
                if self.block_manager.usage_ratio() > watermark and scheduled:
                    break
                self.block_manager.allocate(seq.request_id, num_tokens=needed)
            else:
                target = seq.num_computed_tokens + chunk
                status = self.block_manager.can_append(
                    seq.request_id, extra_tokens=chunk
                )
                if status is AllocStatus.NEVER:
                    seq.mark_finished()
                    finished.append(seq)
                    continue
                if status is AllocStatus.LATER:
                    continue
                self.block_manager.reserve(seq.request_id, target)

            if seq in self.waiting:
                self.waiting.remove(seq)
            if seq not in self.running:
                self.running.append(seq)
            seq.mark_running()
            seq.metadata["_prefill_chunk"] = chunk
            scheduled.append(seq)
            budget -= chunk
            tokens_this_step += chunk

        return SchedulerOutput(
            scheduled=scheduled,
            is_prefill=True,
            num_batched_tokens=tokens_this_step,
            finished=finished,
        )
