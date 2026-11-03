"""Orbit 0.4 — Draft orbits (speculative decode with shallow vs full loops).

Idea
----
Spider's recurrent depth is the expensive part; attention shells write the only
KV state. A *draft orbit* proposes the next K tokens with early_exit=True
(rows leave orbit early → cheaper loops). A *verify orbit* replays the same
tokens with early_exit=False (full loop depth). Matching prefix is accepted;
on the first mismatch we keep the verified token and resync.

This is same-model speculative decoding specialized to Spider loop depth,
not a separate draft network.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spider_moe.model import SpiderKVCache, SpiderWebLM


@dataclass
class DraftOrbitConfig:
    """Speculation knobs for Orbit 0.4."""

    draft_tokens: int = 4
    """Max tokens proposed per draft round."""
    temperature: float = 0.0
    """0 → greedy argmax (recommended for high accept rate)."""
    top_p: float = 1.0
    max_rounds: int | None = None
    """Optional cap on draft/verify rounds (None = until max_new_tokens)."""


@dataclass
class DraftOrbitStats:
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    draft_steps: int = 0
    verify_steps: int = 0

    @property
    def accept_rate(self) -> float:
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """logits: [B, V] → [B, 1] token ids."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / max(float(temperature), 1e-5)
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(scaled.softmax(-1), 1)


@torch.no_grad()
def draft_orbit_generate(
    model: SpiderWebLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cfg: DraftOrbitConfig | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, DraftOrbitStats]:
    """Autoregressive generate with draft-orbit speculation.

    Parameters
    ----------
    model:
        Sentinel / Spider weights (eval mode recommended).
    input_ids:
        [B, T] prompt tokens (B=1 fully supported; B>1 works if all rows share fate).
    max_new_tokens:
        Total new tokens to produce.
    cfg:
        DraftOrbitConfig.
    eos_token_id:
        Optional stop token; stops when *all* batch rows emit EOS.

    Returns
    -------
    output_ids : [B, T + generated]
    stats : DraftOrbitStats
    """
    cfg = cfg or DraftOrbitConfig()
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, seq]")
    if max_new_tokens < 1:
        return input_ids, DraftOrbitStats()

    model.eval()
    stats = DraftOrbitStats()
    output = input_ids
    logits, cache = model.prefill(input_ids)
    generated = 0

    while generated < max_new_tokens:
        stats.rounds += 1
        if cfg.max_rounds is not None and stats.rounds > cfg.max_rounds:
            break

        k = min(cfg.draft_tokens, max_new_tokens - generated)
        base_len = cache.length
        # Logits for the token *before* this draft round (prefill or last verify).
        anchor_logits = logits[:, -1, :].clone()

        # ---- Draft orbit (early exit / shallow) ----
        draft_tokens: list[torch.Tensor] = []
        next_logits = anchor_logits
        for _ in range(k):
            tok = _sample_next(next_logits, cfg.temperature, cfg.top_p)
            draft_tokens.append(tok)
            stats.draft_steps += 1
            stats.proposed += int(tok.shape[0])
            next_logits, cache = model.decode_step(tok, cache, early_exit=True)
            next_logits = next_logits[:, -1, :]
            if eos_token_id is not None and torch.all(tok.squeeze(-1) == eos_token_id):
                break

        if not draft_tokens:
            break

        # ---- Verify orbit (full depth) with KV length rollback ----
        cache.length = base_len
        accepted: list[torch.Tensor] = []
        step_logits = anchor_logits

        for draft_tok in draft_tokens:
            verified = _sample_next(step_logits, cfg.temperature, cfg.top_p)
            stats.verify_steps += 1
            if not torch.equal(verified, draft_tok):
                # Accept verified token; end this speculation round.
                accepted.append(verified)
                stats.accepted += int(verified.shape[0])
                step_logits, cache = model.decode_step(
                    verified, cache, early_exit=False
                )
                step_logits = step_logits[:, -1, :]
                logits = step_logits.unsqueeze(1)
                break
            accepted.append(draft_tok)
            stats.accepted += int(draft_tok.shape[0])
            step_logits, cache = model.decode_step(
                draft_tok, cache, early_exit=False
            )
            step_logits = step_logits[:, -1, :]
            logits = step_logits.unsqueeze(1)

        if not accepted:
            break

        block = torch.cat(accepted, dim=1)
        output = torch.cat([output, block], dim=1)
        generated += block.shape[1]

        if eos_token_id is not None and torch.all(block[:, -1] == eos_token_id):
            break

    return output, stats


@torch.no_grad()
def draft_orbit_accept_rate_probe(
    model: SpiderWebLM,
    input_ids: torch.Tensor,
    draft_tokens: int = 4,
    rounds: int = 4,
) -> DraftOrbitStats:
    """Run a few speculative rounds and return accept stats (greedy)."""
    _, stats = draft_orbit_generate(
        model,
        input_ids,
        max_new_tokens=draft_tokens * rounds,
        cfg=DraftOrbitConfig(draft_tokens=draft_tokens, temperature=0.0),
    )
    return stats
