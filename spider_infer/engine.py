"""SpiderInfer LLMEngine — continuous-batching inference entry point."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM

from .checkpoint import load_checkpoint
from .config import EngineConfig
from .memory.block_manager import BlockManager
from .model_runner import ModelRunner
from .prefix_cache import PrefixCache
from .sampler import Sampler
from .sampling_params import SamplingParams
from .scheduler.scheduler import Scheduler
from .sequence import Sequence, SequenceStatus
from .tokenizer_utils import TokenizerProtocol, load_tokenizer


class RequestOutput:
    """Result of one finished request."""

    def __init__(self, sequence: Sequence, text: str | None = None):
        self.request_id = sequence.request_id
        self.prompt_token_ids = list(sequence.prompt_token_ids)
        self.output_token_ids = list(sequence.output_token_ids)
        self.finished = sequence.is_finished
        self.metadata = dict(sequence.metadata)
        self.text = text

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


class LLMEngine:
    """High-level inference engine with continuous batching + prefix cache."""

    def __init__(
        self,
        model: SpiderWebLM,
        engine_config: EngineConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
        *,
        enable_prefix_cache: bool = True,
        prefix_cache_size: int = 128,
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        if engine_config is None:
            engine_config = EngineConfig.default_for_model(
                max_seq_len=model.cfg.max_seq_len,
                device=str(next(model.parameters()).device),
            )
        self.config = engine_config
        max_model_len = engine_config.max_model_len or model.cfg.max_seq_len

        self.block_manager = BlockManager(
            model, engine_config.cache, device=engine_config.device
        )
        self.scheduler = Scheduler(
            self.block_manager,
            engine_config.scheduler,
            max_model_len=max_model_len,
        )
        self.model_runner = ModelRunner(
            model, self.block_manager,
            enable_cuda_graph=getattr(engine_config, 'enable_cuda_graph', True),
        )
        self.sampler = Sampler()
        self.eos_token_id = engine_config.eos_token_id
        if self.eos_token_id is None and tokenizer is not None:
            self.eos_token_id = tokenizer.eos_token_id

        self._generators: dict[str, torch.Generator] = {}
        self.prefix_cache = PrefixCache(max_entries=prefix_cache_size) if enable_prefix_cache else None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        engine_config: EngineConfig | None = None,
        device: str | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> "LLMEngine":
        device = device or (engine_config.device if engine_config else "cpu")
        model = SpiderWebLM(model_config).to(device)
        if engine_config is None:
            engine_config = EngineConfig.default_for_model(
                max_seq_len=model_config.max_seq_len, device=device
            )
        return cls(model, engine_config, tokenizer=tokenizer)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        engine_config: EngineConfig | None = None,
        device: str | None = None,
        tokenizer_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "LLMEngine":
        device = device or (engine_config.device if engine_config else "cpu")
        model, cfg, _ = load_checkpoint(checkpoint, device=device)
        if engine_config is None:
            engine_config = EngineConfig.default_for_model(
                max_seq_len=cfg.max_seq_len, device=str(device)
            )
        tok = load_tokenizer(tokenizer_path)
        if tok is not None and tok.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"tokenizer vocab {tok.vocab_size} != checkpoint vocab {cfg.vocab_size}"
            )
        return cls(model, engine_config, tokenizer=tok, **kwargs)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: str | list[str] | list[list[int]] | list[int],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
    ) -> list[RequestOutput]:
        token_lists = self._normalize_prompts(prompts, prompt_token_ids)
        params_list = self._normalize_params(sampling_params, len(token_lists))

        for tokens, params in zip(token_lists, params_list):
            seq = Sequence(prompt_token_ids=tokens, sampling_params=params)
            self._maybe_set_generator(seq)
            self._admit(seq)

        outputs: list[RequestOutput] = []
        while self.scheduler.has_unfinished():
            step_outputs = self.step()
            outputs.extend(step_outputs)

        if self.tokenizer is not None:
            for out in outputs:
                out.text = self.tokenizer.decode(out.token_ids)
        return outputs

    def generate_stream(
        self,
        prompt: list[int] | str,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[int]:
        token_lists = self._normalize_prompts(prompt, None)
        params = self._normalize_params(sampling_params, 1)[0]
        seq = Sequence(prompt_token_ids=token_lists[0], sampling_params=params)
        self._maybe_set_generator(seq)
        self._admit(seq)

        yielded = 0
        while self.scheduler.has_unfinished():
            self.step()
            while yielded < len(seq.output_token_ids):
                token = seq.output_token_ids[yielded]
                yielded += 1
                yield token
            if seq.is_finished:
                return

    def generate_text(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """Convenience: string in → string out (requires tokenizer)."""
        if self.tokenizer is None:
            raise RuntimeError("generate_text requires a tokenizer")
        if isinstance(prompts, str):
            prompts = [prompts]
        outs = self.generate(prompts, sampling_params)
        return [o.text or "" for o in outs]

    # ------------------------------------------------------------------
    # Admit with optional prefix-cache hit
    # ------------------------------------------------------------------

    def _admit(self, seq: Sequence) -> None:
        """Try prefix-cache hit, else normal admit."""
        if self.prefix_cache is not None:
            hit = self.prefix_cache.lookup(seq.prompt_token_ids)
            if hit is not None and self.block_manager.has_request(hit.source_request_id):
                try:
                    self.block_manager.fork_prefix(
                        hit.source_request_id,
                        seq.request_id,
                        prefix_len=hit.length,
                    )
                    # Prompt already in KV — skip prefill compute.
                    seq.num_computed_tokens = hit.length
                    seq.mark_running()
                    self.scheduler.running.append(seq)
                    seq.metadata["prefix_cache_hit"] = True
                    return
                except Exception:
                    # Fall through to normal path.
                    if self.block_manager.has_request(seq.request_id):
                        self.block_manager.free(seq.request_id)
        self.scheduler.add_request(seq)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        """Run one scheduler → model → sample cycle."""
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.finished:
            for s in scheduler_output.finished:
                if self.prefix_cache is not None:
                    self.prefix_cache.evict_request(s.request_id)
            self.scheduler.free_finished(scheduler_output.finished)

        if not scheduler_output.scheduled:
            return [self._to_output(s) for s in scheduler_output.finished]

        model_output = self.model_runner.execute(scheduler_output)

        ready = []
        ready_params = []
        ready_gens = []
        for seq in model_output.sequences:
            if scheduler_output.is_prefill and seq.needs_prefill:
                continue
            ready.append(seq)
            ready_params.append(seq.sampling_params)
            ready_gens.append(self._generators.get(seq.request_id))

        newly_finished: list[Sequence] = list(scheduler_output.finished)
        if ready:
            idx = [model_output.sequences.index(s) for s in ready]
            logits = model_output.logits[idx]
            next_tokens = self.sampler.sample(logits, ready_params, ready_gens)
            for seq, token in zip(ready, next_tokens.tolist()):
                seq.append_token(token)
                if seq.check_stop(self.eos_token_id):
                    seq.mark_finished()
                    newly_finished.append(seq)

        # Retain prompt KV in the prefix cache so later identical prompts
        # can fork_prefix instead of recomputing. We copy pages to a stable
        # holder id before freeing the live request.
        if self.prefix_cache is not None:
            for seq in list(model_output.sequences) + list(newly_finished):
                if not (
                    not seq.needs_prefill
                    and seq.num_prompt_tokens > 0
                    and self.block_manager.has_request(seq.request_id)
                ):
                    continue
                holder = f"prefix://{hash(tuple(seq.prompt_token_ids)) & 0xFFFFFFFF:x}"
                try:
                    if not self.block_manager.has_request(holder):
                        self.block_manager.fork_prefix(
                            seq.request_id, holder, prefix_len=seq.num_prompt_tokens
                        )
                    self.prefix_cache.insert(holder, seq.prompt_token_ids, seq.num_prompt_tokens)
                except Exception:
                    pass

        if newly_finished:
            self.scheduler.free_finished(newly_finished)

        return [self._to_output(s) for s in newly_finished]

    def _to_output(self, seq: Sequence) -> RequestOutput:
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(seq.prompt_token_ids + seq.output_token_ids)
        return RequestOutput(seq, text=text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_prompts(
        self,
        prompts: str | list[str] | list[list[int]] | list[int],
        prompt_token_ids: list[list[int]] | None,
    ) -> list[list[int]]:
        if prompt_token_ids is not None:
            return [list(x) for x in prompt_token_ids]

        if isinstance(prompts, str):
            if self.tokenizer is None:
                raise TypeError(
                    "string prompts require a tokenizer; pass token ids or "
                    "construct the engine with tokenizer_path / tokenizer="
                )
            return [self.tokenizer.encode(prompts)]

        if not prompts:
            raise ValueError("prompts must be non-empty")

        # list[int] → single request
        if isinstance(prompts[0], int):
            return [list(prompts)]  # type: ignore[arg-type]

        # list[str]
        if isinstance(prompts[0], str):
            if self.tokenizer is None:
                raise TypeError("string prompts require a tokenizer")
            return [self.tokenizer.encode(p) for p in prompts]  # type: ignore[arg-type]

        # list[list[int]]
        if isinstance(prompts[0], (list, tuple)):
            return [list(p) for p in prompts]  # type: ignore[arg-type]

        raise TypeError(
            "Unsupported prompt type. Expected str, list[str], list[int], or list[list[int]]."
        )

    def _normalize_params(
        self,
        sampling_params: SamplingParams | list[SamplingParams] | None,
        n: int,
    ) -> list[SamplingParams]:
        if sampling_params is None:
            return [SamplingParams() for _ in range(n)]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in range(n)]
        if len(sampling_params) != n:
            raise ValueError("sampling_params length must match number of prompts")
        return list(sampling_params)

    def _maybe_set_generator(self, seq: Sequence) -> None:
        seed = seq.sampling_params.seed
        if seed is None:
            return
        gen = torch.Generator(device="cpu")
        # Multinomial on GPU still works with CPU generator in recent PyTorch;
        # keep CPU generator for portability.
        gen.manual_seed(int(seed))
        self._generators[seq.request_id] = gen

    def stats(self) -> dict[str, Any]:
        return {
            "cache": self.block_manager.stats(),
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "prefix_cache_entries": len(self.prefix_cache) if self.prefix_cache else 0,
        }
