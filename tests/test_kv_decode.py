import torch

from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def tiny():
    return ModelConfig(
        vocab_size=97, hidden_size=32, num_heads=4,
        num_experts=4, top_k=2, edges_per_expert=2, num_loops=2,
        num_prelude_layers=2, num_coda_layers=2, expert_ffn_size=64,
        loop_embed_size=8, max_seq_len=16, dropout=0.0,
    )


def test_prefill_logits_match_full_forward():
    torch.manual_seed(11)
    model = SpiderWebLM(tiny()).eval()
    tokens = torch.randint(0, 97, (2, 7))
    with torch.no_grad():
        full = model(tokens, early_exit=False)["logits"]
        cached, cache = model.prefill(tokens)
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)
    assert cache.length == tokens.shape[1]
    assert len(cache.layers) == tiny().num_prelude_layers + tiny().num_coda_layers


def test_incremental_decode_matches_recomputed_context():
    torch.manual_seed(12)
    model = SpiderWebLM(tiny()).eval()
    prompt = torch.randint(0, 97, (2, 5))
    new_tokens = torch.randint(0, 97, (2, 4))
    with torch.no_grad():
        _, cache = model.prefill(prompt)
        for index in range(new_tokens.shape[1]):
            token = new_tokens[:, index:index+1]
            cached_logits, cache = model.decode_step(token, cache, early_exit=False)
            full_context = torch.cat([prompt, new_tokens[:, :index+1]], dim=1)
            full_logits = model(full_context, early_exit=False)["logits"][:, -1:, :]
            torch.testing.assert_close(
                cached_logits, full_logits, rtol=2e-5, atol=2e-5
            )
    assert cache.length == prompt.shape[1] + new_tokens.shape[1]


def test_chunked_decode_uses_offset_causal_mask():
    torch.manual_seed(13)
    model = SpiderWebLM(tiny()).eval()
    prompt = torch.randint(0, 97, (1, 4))
    chunk = torch.randint(0, 97, (1, 3))
    with torch.no_grad():
        _, cache = model.prefill(prompt)
        cached, cache = model.decode_step(chunk, cache, early_exit=False)
        full = model(torch.cat([prompt, chunk], dim=1), early_exit=False)["logits"][:, -3:, :]
    torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-5)
    assert cache.length == 7


def test_cache_capacity_is_checked():
    model = SpiderWebLM(tiny()).eval()
    tokens = torch.randint(0, 97, (1, 16))
    with torch.no_grad():
        _, cache = model.prefill(tokens)
        try:
            model.decode_step(torch.ones((1, 1), dtype=torch.long), cache)
        except ValueError as exc:
            assert "capacity" in str(exc)
        else:
            raise AssertionError("expected cache capacity error")


def test_generate_uses_sliding_reprefill_at_context_limit():
    torch.manual_seed(14)
    model = SpiderWebLM(tiny()).eval()
    prompt = torch.randint(0, 97, (1, 15))
    generated = model.generate(prompt, max_new_tokens=4, early_exit=False)
    assert generated.shape == (1, 19)


def test_incremental_decode_defaults_to_full_recurrent_depth():
    """Untrained exit_head must not silently shorten inference depth."""
    torch.manual_seed(15)
    model = SpiderWebLM(tiny()).eval()
    # Force the exit classifier to request exit after the first recurrent loop.
    with torch.no_grad():
        model.recurrent.exit_head.weight.zero_()
        model.recurrent.exit_head.bias.fill_(20.0)
    prompt = torch.randint(0, 97, (1, 4))
    token = torch.randint(0, 97, (1, 1))
    with torch.no_grad():
        _, cache = model.prefill(prompt)
        got, _ = model.decode_step(token, cache)  # default must be full-depth
        expected = model(torch.cat([prompt, token], dim=1), early_exit=False)["logits"][:, -1:, :]
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
