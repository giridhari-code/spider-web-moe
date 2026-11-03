"""Minimal dense Transformer LM for Orbit multi-architecture serve.

Not a production Llama replacement — a correct, small dense stack so the
same continuous-batching / server path can target architecture='dense'
without vLLM. Single-GPU first; TP hooks reserved for later.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DenseConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 4
    ffn_size: int = 128
    max_seq_len: int = 256
    dropout: float = 0.0

    def validate(self) -> "DenseConfig":
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        return self


class _DenseBlock(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        h, n = cfg.hidden_size, cfg.num_heads
        self.ln1 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, n, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(h)
        self.ff = nn.Sequential(
            nn.Linear(h, cfg.ffn_size),
            nn.GELU(),
            nn.Linear(cfg.ffn_size, h),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class DenseLM(nn.Module):
    """Small causal dense LM with generate() compatible enough for offline serve demos."""

    def __init__(self, cfg: DenseConfig | None = None):
        super().__init__()
        self.cfg = (cfg or DenseConfig()).validate()
        c = self.cfg
        self.tok = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos = nn.Embedding(c.max_seq_len, c.hidden_size)
        self.blocks = nn.ModuleList([_DenseBlock(c) for _ in range(c.num_layers)])
        self.ln_f = nn.LayerNorm(c.hidden_size)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # MultiheadAttention expects float mask with -inf for blocked positions
        m = torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
        mask = torch.zeros(t, t, device=device)
        mask = mask.masked_fill(m, float("-inf"))
        return mask

    def forward(self, input_ids: torch.Tensor) -> dict:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)
        mask = self._causal_mask(t, input_ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.lm_head(self.ln_f(x))
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.max_seq_len :]
            logits = self.forward(ctx)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                next_id = torch.multinomial((logits / temperature).softmax(-1), 1)
            out = torch.cat([out, next_id], dim=1)
        return out
