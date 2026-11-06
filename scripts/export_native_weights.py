#!/usr/bin/env python3
"""Export SpiderWebLM checkpoint → .spiderw (Phase 1–2.3).

Usage:
  python scripts/export_native_weights.py --checkpoint ckpt.pt --out model.spiderw
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np


def _tensor(t):
    return t.detach().cpu().float().numpy()


def _export_transformer_block(state, tensors, prefix_src, prefix_dst, n_layers, H):
    """prelude/coda: attn.qkv, attn.proj, ln, ffn."""
    for i in range(n_layers):
        base = f"{prefix_src}.{i}"
        dst = f"{prefix_dst}.{i}"
        # CausalSelfAttention: qkv, proj
        for src_k, dst_k in [
            (f"{base}.attn.qkv.weight", f"{dst}.W_qkv"),
            (f"{base}.attn.proj.weight", f"{dst}.W_o"),
            (f"{base}.ln1.weight", f"{dst}.ln1_w"),
            (f"{base}.ln1.bias", f"{dst}.ln1_b"),
            (f"{base}.ln2.weight", f"{dst}.ln2_w"),
            (f"{base}.ln2.bias", f"{dst}.ln2_b"),
        ]:
            if src_k in state:
                tensors[dst_k] = _tensor(state[src_k])
        # FFN: Sequential Linear, GELU, Linear — indices 0 and 2
        up_k = f"{base}.ffn.0.weight"
        down_k = f"{base}.ffn.2.weight"
        if up_k in state and down_k in state:
            tensors[f"{dst}.W_ffn_up"] = _tensor(state[up_k])
            tensors[f"{dst}.W_ffn_down"] = _tensor(state[down_k])


def export_from_state_dict(state: dict, cfg, out_path: str, build_csr: bool = False) -> None:
    from spider_infer.orbit.weight_loader import SpiderWMeta, write_spiderw

    tensors = {}

    def grab(key, name, transform=None):
        if key in state:
            arr = _tensor(state[key])
            if transform:
                arr = transform(arr)
            tensors[name] = arr

    grab("token_embedding.weight", "token_emb")
    grab("tok_emb.weight", "token_emb")
    grab("lm_head.weight", "lm_head")

    grab("recurrent.router.weight", "router")
    if "router" in tensors and tensors["router"].shape[0] != cfg.hidden_size:
        tensors["router"] = tensors["router"].T

    grab("recurrent.loop_bias", "loop_bias")
    grab("recurrent.loop_embedding.weight", "loop_embedding")
    grab("recurrent.edge_generator.weight", "edge_generator")

    # edge_value ModuleList
    E = int(cfg.num_experts)
    H = int(cfg.hidden_size)
    U = int(cfg.expert_ffn_size)
    edge_ws = []
    for e in range(E):
        k = f"recurrent.edge_value.{e}.weight"
        if k in state:
            edge_ws.append(_tensor(state[k]))
    if edge_ws:
        tensors["edge_value"] = np.stack(edge_ws, axis=0)

    ups, downs = [], []
    for e in range(E):
        k0 = f"recurrent.experts.{e}.0.weight"
        k2 = f"recurrent.experts.{e}.2.weight"
        if k0 in state and k2 in state:
            ups.append(_tensor(state[k0]).T)
            downs.append(_tensor(state[k2]).T)
    if ups:
        tensors["expert_up"] = np.stack([u.reshape(H, U) for u in ups], axis=0)
        tensors["expert_down"] = np.stack([d.reshape(U, H) for d in downs], axis=0)

    grab("recurrent.norm.weight", "moe_norm")
    if "recurrent.norm.bias" in state and "moe_norm" in tensors:
        tensors["moe_norm"] = np.stack(
            [_tensor(state["recurrent.norm.weight"]), _tensor(state["recurrent.norm.bias"])],
            axis=0,
        )
    grab("recurrent.exit_head.weight", "exit_head")

    n_pre = int(getattr(cfg, "num_prelude_layers", 0) or 0)
    n_coda = int(getattr(cfg, "num_coda_layers", 0) or 0)
    # detect if not set
    if n_pre == 0:
        while f"prelude.{n_pre}.attn.qkv.weight" in state:
            n_pre += 1
    if n_coda == 0:
        while f"coda.{n_coda}.attn.qkv.weight" in state:
            n_coda += 1

    _export_transformer_block(state, tensors, "prelude", "prelude", n_pre, H)
    _export_transformer_block(state, tensors, "coda", "coda", n_coda, H)

    flags = 0
    if "token_emb" in tensors and (
        "lm_head" not in tensors
        or np.allclose(tensors.get("lm_head", 0), tensors["token_emb"])
    ):
        flags |= 4
        tensors.pop("lm_head", None)

    meta = SpiderWMeta(
        version=1,
        flags=flags,
        vocab_size=int(
            getattr(cfg, "vocab_size", tensors.get("token_emb", np.zeros((1, 1))).shape[0])
        ),
        hidden_size=int(H),
        num_heads=int(getattr(cfg, "num_heads", 8)),
        num_experts=int(E),
        top_k=int(getattr(cfg, "top_k", 2)),
        edges_per_expert=int(getattr(cfg, "edges_per_expert", 2)),
        num_loops=int(getattr(cfg, "num_loops", 2)),
        expert_ffn_size=int(U),
        num_prelude_layers=n_pre,
        num_coda_layers=n_coda,
    )

    if build_csr:
        M = meta.edges_per_expert
        row_ptr = [0]
        col, ew = [], []
        for src in range(E):
            for m in range(M):
                col.append((src + 1 + m) % E)
                ew.append(1.0 / M)
            row_ptr.append(len(col))
        tensors["csr_row_ptr"] = np.asarray(row_ptr, dtype=np.int32)
        tensors["csr_col_idx"] = np.asarray(col, dtype=np.int32)
        tensors["csr_edge_w"] = np.asarray(ew, dtype=np.float32)
        meta.flags |= 2

    write_spiderw(out_path, meta, tensors)
    print(f"wrote {out_path} sections={list(tensors.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-csr-placeholder", action="store_true")
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    cfg = ckpt.get("config")
    if cfg is None:
        emb = state.get("token_embedding.weight", state.get("tok_emb.weight"))
        H = int(emb.shape[1])
        cfg = SimpleNamespace(
            vocab_size=int(emb.shape[0]),
            hidden_size=H,
            num_heads=8,
            num_experts=4,
            top_k=2,
            edges_per_expert=2,
            num_loops=2,
            expert_ffn_size=H * 2,
            num_prelude_layers=0,
            num_coda_layers=0,
        )
        e = 0
        while f"recurrent.experts.{e}.0.weight" in state:
            e += 1
        if e:
            cfg.num_experts = e
            cfg.expert_ffn_size = int(state[f"recurrent.experts.0.0.weight"].shape[0])

    export_from_state_dict(state, cfg, args.out, build_csr=args.build_csr_placeholder)


if __name__ == "__main__":
    main()
