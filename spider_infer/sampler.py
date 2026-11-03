"""Token sampling utilities (temperature, top-p, top-k, multinomial)."""

from __future__ import annotations

import torch

from .sampling_params import SamplingParams


class Sampler:
    """Phase-1 sampler. One set of params applied per batch row."""

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        params_list: list[SamplingParams],
        generators: list[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, vocab]
            params_list: length == batch
            generators: optional per-row RNG

        Returns:
            next_token_ids: [batch] int64
        """
        if logits.ndim != 2:
            raise ValueError("logits must be [batch, vocab]")
        batch = logits.shape[0]
        if len(params_list) != batch:
            raise ValueError("params_list length must equal batch size")

        next_ids = []
        for i in range(batch):
            row = logits[i]
            params = params_list[i]
            gen = None if generators is None else generators[i]

            # Greedy fast path
            if params.temperature == 0.0:
                next_ids.append(int(torch.argmax(row).item()))
                continue

            row = row / max(params.temperature, 1e-5)

            if params.top_k > 0:
                top_k = min(params.top_k, row.numel())
                values, indices = torch.topk(row, top_k)
                filtered = torch.full_like(row, float("-inf"))
                filtered.scatter_(0, indices, values)
                row = filtered

            if params.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                # Keep the first token that crosses top_p as well.
                mask = cumulative - probs > params.top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                filtered = torch.full_like(row, float("-inf"))
                filtered.scatter_(0, sorted_idx, sorted_logits)
                row = filtered

            probs = torch.softmax(row, dim=-1)
            if gen is not None:
                token = torch.multinomial(probs, num_samples=1, generator=gen)
            else:
                token = torch.multinomial(probs, num_samples=1)
            next_ids.append(int(token.item()))

        return torch.tensor(next_ids, dtype=torch.long, device=logits.device)
