"""Runs prefill / extend / decode against SpiderWebLM using the paged cache."""

from __future__ import annotations

from .memory.workspace import GLOBAL_WORKSPACE

from dataclasses import dataclass

import torch

from spider_moe.model import SpiderWebLM

from .memory.block_manager import BlockManager
from .cuda_graph import CudaGraphDecodeRunner
from .scheduler.scheduler import SchedulerOutput
from .sequence import Sequence


@dataclass
class ModelOutput:
    """Logits for the last position of every scheduled sequence."""

    logits: torch.Tensor  # [batch, vocab]
    sequences: list[Sequence]


class ModelRunner:
    """Adapter between the scheduler and SpiderWebLM paged APIs."""

    def __init__(
        self,
        model: SpiderWebLM,
        block_manager: BlockManager,
        enable_cuda_graph: bool = True,
    ):
        self.model = model
        self.block_manager = block_manager
        self.cache = block_manager.cache
        self.cuda_graph = CudaGraphDecodeRunner(enabled=enable_cuda_graph)

    @torch.inference_mode()
    def execute(self, scheduler_output: SchedulerOutput) -> ModelOutput:
        if not scheduler_output.scheduled:
            raise RuntimeError("ModelRunner.execute called with empty schedule")

        if scheduler_output.is_prefill:
            return self._prefill_or_extend(scheduler_output.scheduled)
        return self._decode(scheduler_output.scheduled)

    def _prefill_or_extend(self, sequences: list[Sequence]) -> ModelOutput:
        request_ids = [s.request_id for s in sequences]

        token_lists: list[list[int]] = []
        for seq in sequences:
            remaining = seq.prompt_token_ids[seq.num_computed_tokens :]
            chunk = int(seq.metadata.get("_prefill_chunk", len(remaining)))
            chunk = max(1, min(chunk, len(remaining)))
            token_lists.append(remaining[:chunk])

        max_len = max(len(t) for t in token_lists)
        if max_len == 0:
            raise RuntimeError("prefill scheduled but no tokens remain")

        batch = len(sequences)
        input_ids = torch.zeros(
            (batch, max_len), dtype=torch.long, device=self.cache.device
        )
        for i, tokens in enumerate(token_lists):
            input_ids[i, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=self.cache.device
            )

        lengths = [self.cache.sequence_length(rid) for rid in request_ids]
        if len(set(lengths)) != 1:
            raise RuntimeError(
                f"prefill/extend batch has unequal cache lengths: {lengths}"
            )
        cache_len = lengths[0]

        if cache_len == 0:
            if len(set(len(t) for t in token_lists)) != 1:
                return self._run_unequal(
                    sequences, token_lists, request_ids, mode="prefill"
                )
            logits, _ = self.model.prefill_paged(
                input_ids, self.cache, request_ids=request_ids
            )
        else:
            if len(set(len(t) for t in token_lists)) != 1:
                return self._run_unequal(
                    sequences, token_lists, request_ids, mode="extend"
                )
            logits, _ = self.model.decode_step_paged(
                input_ids, self.cache, request_ids, early_exit=False
            )

        for i, seq in enumerate(sequences):
            seq.num_computed_tokens += len(token_lists[i])
            seq.metadata.pop("_prefill_chunk", None)

        last_logits = [logits[i, len(tokens) - 1, :] for i, tokens in enumerate(token_lists)]
        return ModelOutput(logits=torch.stack(last_logits, dim=0), sequences=sequences)

    def _run_unequal(
        self,
        sequences: list[Sequence],
        token_lists: list[list[int]],
        request_ids: list,
        mode: str,
    ) -> ModelOutput:
        last_logits = []
        for seq, tokens, rid in zip(sequences, token_lists, request_ids):
            ids = torch.tensor([tokens], dtype=torch.long, device=self.cache.device)
            if mode == "prefill":
                logits, _ = self.model.prefill_paged(
                    ids, self.cache, request_ids=[rid]
                )
            else:
                logits, _ = self.model.decode_step_paged(
                    ids, self.cache, [rid], early_exit=False
                )
            last_logits.append(logits[0, len(tokens) - 1, :])
            seq.num_computed_tokens += len(tokens)
            seq.metadata.pop("_prefill_chunk", None)
        return ModelOutput(
            logits=torch.stack(last_logits, dim=0), sequences=sequences
        )

    def _decode(self, sequences: list[Sequence]) -> ModelOutput:
        request_ids = [s.request_id for s in sequences]
        token_ids = []
        for seq in sequences:
            if seq.output_token_ids:
                token_ids.append(seq.output_token_ids[-1])
            else:
                token_ids.append(seq.prompt_token_ids[-1])

        # Workspace pool: reuse [B,1] long buffers across decode steps
        b = len(token_ids)
        buf = GLOBAL_WORKSPACE.acquire(
            (b, 1), dtype=torch.long, device=self.cache.device
        )
        buf[:, 0] = torch.as_tensor(token_ids, dtype=torch.long, device=self.cache.device)
        input_ids = buf

        early_exit = sequences[0].sampling_params.early_exit
        cache_len = self.cache.sequence_length(request_ids[0])
        num_loops = int(getattr(self.model.cfg, "num_loops", 1))

        def _fn(ids, rids):
            logits, _ = self.model.decode_step_paged(
                ids, self.cache, rids, early_exit=early_exit
            )
            return logits

        # Orbit 0.3: CUDA graph only when early_exit is off (static ActiveSet / loops).
        # Key includes (batch, length_bucket, num_loops).
        if not early_exit:
            logits = self.cuda_graph.run(
                _fn, input_ids, request_ids, cache_len, num_loops=num_loops
            )
        else:
            logits = _fn(input_ids, request_ids)

        for seq in sequences:
            seq.num_computed_tokens += 1

        out = ModelOutput(logits=logits[:, -1, :], sequences=sequences)
        GLOBAL_WORKSPACE.release(input_ids)
        return out
