"""Phase A.2 — numerical parity harness for sparse expert path.

Covers:
1. sparse_expert_delta vs dense expert stack (exact equivalence)
2. Full SpiderMoELayer forward is finite and shape-correct
3. Prefill logits match full forward (regression guard)
4. Incremental decode matches recomputed context (regression guard)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.sparse_experts import sparse_expert_delta


def _tiny(**kwargs) -> ModelConfig:
    cfg = dict(
        vocab_size=97,
        hidden_size=32,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=2,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=32,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
    )
    cfg.update(kwargs)
    return ModelConfig(**cfg).validate()


def _dense_expert_delta(messages, top_indices, top_weights, w_up, b_up, w_down, b_down):
    """Reference dense path: evaluate all experts then mask by router weights."""
    b, t, width = messages.shape
    e = w_up.shape[0]
    up = torch.einsum("bth,eoh->bteo", messages, w_up) + b_up.view(1, 1, e, -1)
    up = F.gelu(up)
    expert_stack = torch.einsum("bteu,eou->bteo", up, w_down) + b_down.view(1, 1, e, -1)
    expert_weight = messages.new_zeros(b, t, e)
    expert_weight.scatter_add_(-1, top_indices, top_weights)
    return (expert_stack * expert_weight.unsqueeze(-1)).sum(dim=2)


def test_sparse_expert_delta_matches_dense():
    torch.manual_seed(42)
    b, t, h, e, k, h_up = 3, 5, 32, 4, 2, 64
    messages = torch.randn(b, t, h)
    # Random top-k indices + normalized weights
    scores = torch.randn(b, t, e)
    top_weights, top_indices = scores.softmax(-1).topk(k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    w_up = torch.randn(e, h_up, h)
    b_up = torch.randn(e, h_up)
    w_down = torch.randn(e, h, h_up)
    b_down = torch.randn(e, h)

    sparse = sparse_expert_delta(
        messages, top_indices, top_weights,
        w_up, b_up, w_down, b_down,
        num_experts=e, top_k=k, expert_parallel=False,
    )
    dense = _dense_expert_delta(
        messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
    )
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)


def test_sparse_expert_delta_handles_duplicate_expert_slots():
    """Same expert selected in multiple top-k slots must accumulate weights."""
    torch.manual_seed(7)
    b, t, h, e, h_up = 2, 3, 16, 3, 32
    messages = torch.randn(b, t, h)
    # Force expert 1 in both slots for every token
    top_indices = torch.ones(b, t, 2, dtype=torch.long)
    top_weights = torch.tensor([0.4, 0.6]).view(1, 1, 2).expand(b, t, 2).contiguous()
    w_up = torch.randn(e, h_up, h)
    b_up = torch.randn(e, h_up)
    w_down = torch.randn(e, h, h_up)
    b_down = torch.randn(e, h)

    sparse = sparse_expert_delta(
        messages, top_indices, top_weights,
        w_up, b_up, w_down, b_down,
        num_experts=e, top_k=2, expert_parallel=False,
    )
    dense = _dense_expert_delta(
        messages, top_indices, top_weights, w_up, b_up, w_down, b_down,
    )
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)


def test_moe_layer_forward_finite_and_shaped():
    torch.manual_seed(0)
    model = SpiderWebLM(_tiny()).eval()
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        out = model(x, early_exit=False, return_stats=True)
    assert out["logits"].shape == (2, 8, 97)
    assert torch.isfinite(out["logits"]).all()
    assert out["loops_used"] == 2
    assert len(out["stats"]) == 2


def test_prefill_matches_full_forward():
    torch.manual_seed(11)
    model = SpiderWebLM(_tiny()).eval()
    tokens = torch.randint(0, 97, (2, 7))
    with torch.no_grad():
        full = model(tokens, early_exit=False)["logits"]
        cached, cache = model.prefill(tokens)
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)
    assert cache.length == tokens.shape[1]


def test_incremental_decode_matches_full_recompute():
    torch.manual_seed(12)
    model = SpiderWebLM(_tiny()).eval()
    prompt = torch.randint(0, 97, (2, 5))
    new_tokens = torch.randint(0, 97, (2, 4))
    with torch.no_grad():
        _, cache = model.prefill(prompt)
        for index in range(new_tokens.shape[1]):
            token = new_tokens[:, index : index + 1]
            cached_logits, cache = model.decode_step(token, cache, early_exit=False)
            full_context = torch.cat([prompt, new_tokens[:, : index + 1]], dim=1)
            full_logits = model(full_context, early_exit=False)["logits"][:, -1:, :]
            torch.testing.assert_close(
                cached_logits, full_logits, rtol=2e-5, atol=2e-5
            )
    assert cache.length == prompt.shape[1] + new_tokens.shape[1]


def test_sparse_path_backward_reaches_router():
    """Training still flows gradients through the sparse expert path."""
    torch.manual_seed(3)
    model = SpiderWebLM(_tiny())
    x = torch.randint(0, 97, (2, 6))
    loss = model(x, targets=x)["loss"]
    loss.backward()
    assert model.recurrent.router.weight.grad is not None
    assert torch.isfinite(model.recurrent.router.weight.grad).all()
    # At least one expert weight should receive grad
    grads = [ex[0].weight.grad for ex in model.recurrent.experts]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
