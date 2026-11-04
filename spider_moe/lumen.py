"""Lumen — Spider global workspace (inspired by Global Workspace Theory / J-space).

Anthropic (2026) reports a privileged *J-space* in language models: a small set of
internal patterns that are reportable, controllable, used for multi-step reasoning,
and broadcast-capable — matching functional properties of Baars/Dehaene global
workspace theory — without claiming consciousness.

Lumen is the *architectural* analogue for Spider/Sentinel:

  specialists  = experts / loop modules (parallel, local)
  workspace    = low-rank shared slots (the “global workspace”)
  broadcast    = gated write-back into the residual stream
  j-lens       = optional projection of workspace → vocab (readout of “what’s on mind”)

This is not a copy of Anthropic’s Jacobian-lens methodology on Claude; it is a
trainable / interpretable workspace block that Orbit can host and inspect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LumenState:
    """Workspace contents after a write cycle."""

    slots: torch.Tensor          # [B, T, S, D] or [B, T, S, slot_dim]
    slot_weights: torch.Tensor   # [B, T, S] competition scores
    broadcast: torch.Tensor      # [B, T, H] residual contribution
    j_logits: torch.Tensor | None = None  # [B, T, V] optional readout


class LumenWorkspace(nn.Module):
    """Low-rank global workspace with competitive write + broadcast.

    Parameters
    ----------
    hidden_size:
        Model width H.
    num_slots:
        Workspace capacity S (small — privileged channel).
    slot_dim:
        Per-slot rank; defaults to hidden_size // 4.
    vocab_size:
        If set, enables J-lens readout head (workspace → vocab).
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 8,
        slot_dim: int | None = None,
        vocab_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim or max(32, hidden_size // 4))
        self.vocab_size = vocab_size

        self.write_q = nn.Linear(hidden_size, self.num_slots, bias=False)
        self.write_v = nn.Linear(hidden_size, self.slot_dim, bias=False)
        self.slot_to_h = nn.Linear(self.slot_dim, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size + hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        self.j_lens: nn.Linear | None = None
        if vocab_size is not None and vocab_size > 0:
            self.j_lens = nn.Linear(self.slot_dim, vocab_size, bias=False)

        # Learnable slot priors (what the workspace “prefers” to hold)
        self.slot_bias = nn.Parameter(torch.zeros(self.num_slots))

    def forward(
        self,
        h: torch.Tensor,
        *,
        return_state: bool = False,
        j_lens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LumenState]:
        """Write specialists' residual ``h`` into workspace; broadcast back.

        h: [B, T, H]
        returns updated residual (h + broadcast) or (updated, LumenState)
        """
        b, t, _ = h.shape
        # Competition over slots per token
        scores = self.write_q(h) + self.slot_bias  # [B,T,S]
        weights = F.softmax(scores, dim=-1)
        values = self.write_v(h)  # [B,T,D]
        # slots[s] = sum_t weights[...,s] * values  (token writes into slots)
        slots = torch.einsum("bts,btd->btsd", weights, values)

        # Broadcast: weighted sum of slot contents → H
        mixed = torch.einsum("bts,btsd->btd", weights, slots)
        broadcast_raw = self.slot_to_h(mixed)
        gate = torch.sigmoid(self.gate(torch.cat([h, broadcast_raw], dim=-1)))
        broadcast = self.drop(gate * broadcast_raw)
        out = self.norm(h + broadcast)

        state = None
        if return_state or j_lens:
            j_logits = None
            if j_lens and self.j_lens is not None:
                # Reportable content: average slot vectors → vocab
                slot_mean = slots.mean(dim=2)  # [B,T,D]
                j_logits = self.j_lens(slot_mean)
            state = LumenState(
                slots=slots,
                slot_weights=weights,
                broadcast=broadcast,
                j_logits=j_logits,
            )
        if return_state:
            return out, state
        return out

    def read_j_space(self, state: LumenState, top_k: int = 5) -> list[list[list[int]]]:
        """Return top-k vocab ids per token from J-lens (reportable workspace)."""
        if state.j_logits is None:
            return []
        vals, idx = torch.topk(state.j_logits, k=min(top_k, state.j_logits.shape[-1]), dim=-1)
        return idx.tolist()

    def stats(self) -> dict[str, Any]:
        return {
            "name": "Lumen",
            "analogue": "global_workspace_j_space",
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "j_lens": self.j_lens is not None,
            "ref": "https://www.anthropic.com/research/global-workspace",
        }
