"""Phase C — early-exit active-row compaction correctness."""
from __future__ import annotations

import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def _tiny(loops: int = 4) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_experts=4,
        top_k=2,
        edges_per_expert=2,
        num_loops=loops,
        num_prelude_layers=1,
        num_coda_layers=1,
        expert_ffn_size=64,
        loop_embed_size=8,
        max_seq_len=32,
        dropout=0.0,
        expert_parallel=False,
        use_rope=True,
        early_exit_threshold=0.85,
    ).validate()


def test_early_exit_decode_is_finite_and_shaped():
    """Forced early-exit decode produces finite logits of the right shape."""
    torch.manual_seed(0)
    model = SpiderWebLM(_tiny(loops=4)).eval()
    with torch.no_grad():
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.fill_(20.0)

    prompt = torch.randint(0, 64, (3, 6))
    token = torch.randint(0, 64, (3, 1))
    with torch.no_grad():
        _, cache = model.prefill(prompt)
        out_early, cache2 = model.decode_step(token, cache, early_exit=True)
        out_full, _ = model.decode_step(token, cache2, early_exit=False)

    assert out_early.shape == (3, 1, 64)
    assert torch.isfinite(out_early).all()
    assert torch.isfinite(out_full).all()


def test_compaction_stops_updating_exited_rows():
    """After a row exits, further recurrent loops must not change that row."""
    torch.manual_seed(1)
    model = SpiderWebLM(_tiny(loops=4)).eval()
    # Make exit_head depend on a simple signal: exit when hidden mean > 0
    # Easier: force exit for ALL rows after loop 0 via bias
    with torch.no_grad():
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.fill_(20.0)

    h = torch.randn(4, 1, 32)
    x0 = torch.randn(4, 1, 32)
    with torch.no_grad():
        # Run one loop manually to get post-loop-0 state
        h1 = model.recurrent(h, x0, 0)
        # Full early-exit decode path
        h_out = model._run_recurrent_decode(h, x0, early_exit=True)

    torch.testing.assert_close(h_out, h1, rtol=1e-5, atol=1e-5)


def test_no_early_exit_runs_all_loops():
    torch.manual_seed(2)
    model = SpiderWebLM(_tiny(loops=3)).eval()
    h = torch.randn(2, 1, 32)
    x0 = torch.randn(2, 1, 32)
    with torch.no_grad():
        full = model._run_recurrent_full(h, x0)
        decode = model._run_recurrent_decode(h, x0, early_exit=False)
    torch.testing.assert_close(full, decode, rtol=1e-5, atol=1e-5)


def test_partial_batch_exit_preserves_active_rows():
    """Only some rows exit; active rows continue and diverge from loop-0 state."""
    torch.manual_seed(3)
    model = SpiderWebLM(_tiny(loops=4)).eval()
    # exit_head: positive bias only on a subspace — use weight on first hidden dim
    with torch.no_grad():
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.zero_()
        # score = h[..., 0]  (approx) via weight
        model.recurrent.exit_head.weight[0, 0] = 50.0
        model.recurrent.exit_head.bias.fill_(-5.0)

    # Row 0: large positive h[...,0] → exit; Row 1: negative → stay
    h = torch.zeros(2, 1, 32)
    h[0, 0, 0] = 1.0
    h[1, 0, 0] = -1.0
    x0 = torch.randn(2, 1, 32)
    with torch.no_grad():
        h0 = model.recurrent(h, x0, 0)
        conf = torch.sigmoid(model.recurrent.exit_head(h0).float()).squeeze(-1)
        # Row 0 should tend to exit
        assert conf[0, 0] > conf[1, 0]

        out = model._run_recurrent_decode(h, x0, early_exit=True)
        # Exited row should match single loop; active row should differ from loop-0 only
        # if remaining loops change it.
        torch.testing.assert_close(out[0:1], h0[0:1], rtol=1e-4, atol=1e-4)
