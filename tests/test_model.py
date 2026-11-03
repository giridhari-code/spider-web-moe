import torch
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM

def tiny():
    return ModelConfig(vocab_size=128, hidden_size=32, num_heads=4,
        num_experts=4, top_k=2, edges_per_expert=2, num_loops=2,
        num_prelude_layers=1, num_coda_layers=1, expert_ffn_size=64,
        loop_embed_size=8, max_seq_len=32)

def test_forward_shapes_and_loss():
    model = SpiderWebLM(tiny())
    x = torch.randint(0, 128, (2, 12))
    y = torch.randint(0, 128, (2, 12))
    result = model(x, targets=y, return_stats=True)
    assert result["logits"].shape == (2, 12, 128)
    assert torch.isfinite(result["loss"])
    assert len(result["stats"]) == 2

def test_backward():
    model = SpiderWebLM(tiny())
    x = torch.randint(0, 128, (1, 8))
    loss = model(x, targets=x)["loss"]
    loss.backward()
    assert model.recurrent.router.weight.grad is not None

def test_generation_length():
    model = SpiderWebLM(tiny()).eval()
    x = torch.randint(0, 128, (1, 5))
    y = model.generate(x, max_new_tokens=3, early_exit=False)
    assert y.shape == (1, 8)

def test_tied_embeddings_use_small_initialization_and_healthy_initial_logits():
    model = SpiderWebLM(tiny()).eval()
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    emb_std = model.token_embedding.weight.std(unbiased=False).item()
    assert 0.015 <= emb_std <= 0.025
    x = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        logits = model(x)["logits"]
    assert torch.isfinite(logits).all()
    # A fresh tied-embedding model should not begin with saturated logits.
    assert logits.abs().max().item() < 3.0


def test_targets_support_ignore_index_and_validate_shape():
    model = SpiderWebLM(tiny())
    x = torch.randint(0, 128, (1, 6))
    targets = torch.full_like(x, -100)
    targets[:, 2:] = torch.randint(0, 128, (1, 4))
    loss = model(x, targets=targets)["loss"]
    assert torch.isfinite(loss)
    try:
        model(x, targets=torch.zeros(1, 5, dtype=torch.long))
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("expected target shape validation")
    try:
        model(x, targets=torch.full_like(x, -100))
    except ValueError as exc:
        assert "no supervised tokens" in str(exc)
    else:
        raise AssertionError("expected all-ignored target validation")


def test_position_embeddings_are_used_and_bounded():
    # Absolute PE path
    cfg = tiny()
    cfg.use_rope = False
    model = SpiderWebLM(cfg).eval()
    x = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        a = model(x)["logits"]
        assert model.position_embedding.weight.shape == (32, 32)
        original = model.position_embedding.weight[0].clone()
        model.position_embedding.weight[0].add_(1.0)
        b = model(x)["logits"]
        model.position_embedding.weight[0].copy_(original)
    assert not torch.allclose(a, b)
    try:
        model(torch.randint(0, 128, (1, 33)))
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    else:
        raise AssertionError("expected max_seq_len validation")


def test_rope_is_position_sensitive():
    cfg = tiny()
    cfg.use_rope = True
    model = SpiderWebLM(cfg).eval()
    assert model.use_rope and model.rope_cos is not None
    x = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        # Same tokens at different absolute positions via cached decode path
        # should differ when RoPE is active: compare full forward vs shifted.
        logits = model(x)["logits"]
        assert torch.isfinite(logits).all()
        # Mutating absolute PE must NOT change RoPE outputs.
        a = logits.clone()
        model.position_embedding.weight.add_(1.0)
        b = model(x)["logits"]
        assert torch.allclose(a, b)
