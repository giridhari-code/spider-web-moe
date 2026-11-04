"""Forge 0.4 — DeepEP-style dispatch / combine."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from spider_infer.forge import FORGE_VERSION, ForgeConfig, ForgeBackend, init_forge
from spider_infer.forge.ep.dispatch import (
    build_dispatch_plan,
    dispatch_expert_ffn,
)


def test_forge_version_0_4():
    assert tuple(map(int, FORGE_VERSION.split(".")[:2])) >= (0, 4)


def test_build_dispatch_plan_packs_tokens():
    # 2 tokens, top_k=2, 4 experts
    top_indices = torch.tensor([[[0, 1], [1, 2]]])  # [1,2,2]
    top_weights = torch.tensor([[[0.7, 0.3], [0.6, 0.4]]])
    plan = build_dispatch_plan(top_indices, top_weights, num_experts=4, local_only=False)
    eids = {a.expert_id for a in plan.assignments}
    assert eids == {0, 1, 2}
    a1 = plan.tokens_for(1)
    assert a1 is not None
    assert a1.token_index.numel() == 2  # both tokens routed to expert 1


def test_dispatch_ffn_matches_dense_mix():
    torch.manual_seed(0)
    b, t, h, e, k = 2, 3, 16, 4, 2
    messages = torch.randn(b, t, h)
    logits = torch.randn(b, t, e)
    top_w, top_i = torch.topk(logits, k=k, dim=-1)
    top_w = torch.softmax(top_w, dim=-1)

    w_up = torch.randn(e, 32, h)
    b_up = torch.randn(e, 32)
    w_down = torch.randn(e, h, 32)
    b_down = torch.randn(e, h)

    delta = dispatch_expert_ffn(
        messages, top_i, top_w,
        w_up=w_up, b_up=b_up, w_down=w_down, b_down=b_down,
        num_experts=e,
    )
    assert delta.shape == messages.shape

    # Reference dense mix
    ref = torch.zeros_like(messages)
    flat_m = messages.reshape(-1, h)
    flat_i = top_i.reshape(-1, k)
    flat_w = top_w.reshape(-1, k)
    flat_r = ref.reshape(-1, h)
    for n in range(flat_m.shape[0]):
        for slot in range(k):
            eid = int(flat_i[n, slot])
            w = flat_w[n, slot]
            tok = flat_m[n : n + 1]
            up = F.linear(tok, w_up[eid], b_up[eid])
            out = F.linear(F.gelu(up), w_down[eid], b_down[eid])
            flat_r[n] = flat_r[n] + out[0] * w
    assert torch.allclose(delta, ref, rtol=1e-4, atol=1e-4)


def test_forge_ep_run_dispatched_ffn():
    rt = init_forge(ForgeConfig(device="cpu", backend=ForgeBackend.TORCH))
    b, t, h, e, k = 1, 4, 8, 3, 2
    messages = torch.randn(b, t, h)
    top_i = torch.randint(0, e, (b, t, k))
    top_w = torch.softmax(torch.randn(b, t, k), dim=-1)
    w_up = torch.randn(e, 16, h)
    b_up = torch.zeros(e, 16)
    w_down = torch.randn(e, h, 16)
    b_down = torch.zeros(e, h)
    delta = rt.ep.run_dispatched_ffn(
        messages, top_i, top_w,
        w_up=w_up, b_up=b_up, w_down=w_down, b_down=b_down,
        num_experts=e,
    )
    assert delta.shape == (b, t, h)
    assert rt.ep.stats()["forge_version_target"] == "0.4"
    assert "dispatch" in rt.ep.stats()["last_kernel"]
