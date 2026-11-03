import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .global_paged_kv import GlobalPagedKVCache
from .paged_attention import paged_attention
from .rope import apply_rope, precompute_freqs_cis
from .expert_parallel import all_reduce_sum_, ep_enabled, get_rank_world, owns_expert
from .quant import Int4Tensor, quantize_int4
from .sparse_experts import sparse_expert_delta
from .expert_store import ExpertChunkStore, offload_module_list_to_cpu
from .active_set import ActiveSet, compact_active, scatter_active



@dataclass
class KVLayerCache:
    """Preallocated K/V storage for one attention layer."""
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class SpiderKVCache:
    """Per-attention-layer cache shared across prefill and incremental decode."""
    layers: list[KVLayerCache]
    length: int = 0
    max_seq_len: int = 0

    def reset(self):
        self.length = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, heads, dropout=0.0, use_rope=False, rope_cos=None, rope_sin=None):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.hidden = hidden
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.dropout = dropout
        self.use_rope = use_rope
        # Shared tables registered on the parent model; kept as plain attrs (not parameters).
        self._rope_cos = rope_cos
        self._rope_sin = rope_sin

    def set_rope(self, cos, sin):
        self._rope_cos = cos
        self._rope_sin = sin
        self.use_rope = cos is not None and sin is not None

    def _project(self, x, position: int = 0):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(y):
            return y.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        if self.use_rope and self._rope_cos is not None:
            q = apply_rope(q, self._rope_cos, self._rope_sin, position=position)
            k = apply_rope(k, self._rope_cos, self._rope_sin, position=position)
        return q, k, v

    def forward(self, x, position: int = 0):
        q, k, v = self._project(x, position=position)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(x.shape)
        return self.proj(y)

    def forward_cached(self, x, cache: KVLayerCache, position: int):
        """Attend to cached prefix plus x, writing the new K/V in-place."""
        q, k, v = self._project(x, position=position)
        b, _, t, _ = k.shape
        end = position + t
        if end > cache.keys.shape[2]:
            raise ValueError(
                f"KV cache capacity exceeded: need {end}, capacity {cache.keys.shape[2]}"
            )
        if cache.keys.shape[0] != b:
            raise ValueError("KV cache batch size does not match input batch size")

        # Cache dtype follows the model/cache allocation dtype. This also keeps
        # the K/V tensors compatible with the attention query on mixed precision.
        cache.keys[:, :, position:end].copy_(k.to(cache.keys.dtype))
        cache.values[:, :, position:end].copy_(v.to(cache.values.dtype))
        k_all = cache.keys[:, :, :end]
        v_all = cache.values[:, :, :end]
        q = q.to(k_all.dtype)

        if position == 0:
            # Standard triangular causal attention for the prefill sequence.
            y = F.scaled_dot_product_attention(
                q, k_all, v_all, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )
        elif t == 1:
            # A single decode query can see the entire valid prefix. Avoid
            # materializing a mask so SDPA can select a faster backend.
            y = F.scaled_dot_product_attention(
                q, k_all, v_all, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Offset causal mask for multi-token decode chunks: query position
            # p+i may see keys <= p+i.
            q_positions = position + torch.arange(t, device=x.device)
            k_positions = torch.arange(end, device=x.device)
            allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            y = F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(b, t, self.hidden)
        y = y.to(self.proj.weight.dtype)
        return self.proj(y)

    def forward_paged(self, x, cache: GlobalPagedKVCache, request_ids, layer_index, position):
        q, k, v = self._project(x, position=position)
        cache.write_layer_batch(request_ids, layer_index, position, k.transpose(1, 2), v.transpose(1, 2))
        q = q.to(getattr(cache, 'compute_dtype', cache.dtype))
        y = paged_attention(q, cache, request_ids, layer_index, position)
        y = y.transpose(1, 2).contiguous().view(x.shape)
        y = y.to(self.proj.weight.dtype)
        return self.proj(y)

class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_size
        self.ln1 = nn.LayerNorm(h)
        self.attn = CausalSelfAttention(
            h, cfg.num_heads, cfg.dropout,
            use_rope=getattr(cfg, "use_rope", False),
        )
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, 4*h), nn.GELU(),
            nn.Linear(4*h, h), nn.Dropout(cfg.dropout)
        )

    def forward(self, x, position: int = 0):
        x = x + self.attn(self.ln1(x), position=position)
        return x + self.ffn(self.ln2(x))

    def forward_cached(self, x, cache: KVLayerCache, position: int):
        x = x + self.attn.forward_cached(self.ln1(x), cache, position)
        return x + self.ffn(self.ln2(x))

    def forward_paged(self, x, cache, request_ids, layer_index, position):
        x = x + self.attn.forward_paged(
            self.ln1(x), cache, request_ids, layer_index, position
        )
        return x + self.ffn(self.ln2(x))

class SpiderMoELayer(nn.Module):
    """Shared recurrent layer: token router + per-loop sparse graph + expert FFNs."""
    def __init__(self, cfg):
        super().__init__()
        h, e = cfg.hidden_size, cfg.num_experts
        self.num_experts, self.top_k = e, cfg.top_k
        # top_dobu = GPU chunk LRU (Dobu); independent of router top_k
        self.top_dobu = int(getattr(cfg, "offload_max_gpu_experts", 4))
        self.edges_per_expert = cfg.edges_per_expert
        self.router = nn.Linear(h, e, bias=False)
        self.loop_bias = nn.Parameter(torch.zeros(cfg.num_loops, e))
        self.loop_embedding = nn.Embedding(cfg.num_loops, cfg.loop_embed_size)
        self.edge_generator = nn.Linear(h + cfg.loop_embed_size, e * e)
        self.edge_value = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(e)])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, cfg.expert_ffn_size),
                nn.GELU(),
                nn.Linear(cfg.expert_ffn_size, h)
            ) for _ in range(e)
        ])
        self.norm = nn.LayerNorm(h)
        self.exit_head = nn.Linear(h, 1)
        self.residual_scale = cfg.residual_scale
        self.reinject_weight = cfg.re_inject_weight
        self.expert_parallel = bool(getattr(cfg, "expert_parallel", True))
        self.offload_experts = bool(getattr(cfg, "offload_experts", False))
        self._expert_store: ExpertChunkStore | None = None
        if self.offload_experts:
            # Build CPU chunk catalogue; expert modules themselves leave the GPU.
            compute = next(self.router.parameters()).device
            self.top_dobu = int(getattr(cfg, "offload_max_gpu_experts", 4))
            self._expert_store = ExpertChunkStore(
                self.experts,
                storage_device="cpu",
                compute_device=compute,
                max_gpu_experts=self.top_dobu,
            )
            offload_module_list_to_cpu(self.experts)
        # Optional 4-bit expert weights (populated by quantize_experts()).
        self._int4_w_up: Int4Tensor | None = None
        self._int4_w_down: Int4Tensor | None = None
        self._int4_b_up = None
        self._int4_b_down = None


    @torch.no_grad()
    def quantize_experts(self, group_size: int = 64) -> None:
        """Pack expert up/down weights to group-wise INT4 (inference optimization).

        Biases stay full precision. After this call, forward uses dequantized
        weights; original nn.Linear parameters remain for checkpoint compatibility
        but are not used in the matmul path until dequantize_experts() is called.
        """
        w_up = torch.stack([ex[0].weight.detach() for ex in self.experts], dim=0)
        w_down = torch.stack([ex[2].weight.detach() for ex in self.experts], dim=0)
        self._int4_w_up = quantize_int4(w_up, group_size=group_size)
        self._int4_w_down = quantize_int4(w_down, group_size=group_size)
        self._int4_b_up = torch.stack([ex[0].bias.detach() for ex in self.experts], dim=0)
        self._int4_b_down = torch.stack([ex[2].bias.detach() for ex in self.experts], dim=0)

    def dequantize_experts(self) -> None:
        """Clear INT4 expert buffers and resume using nn.Linear weights."""
        self._int4_w_up = None
        self._int4_w_down = None
        self._int4_b_up = None
        self._int4_b_down = None

    def forward(self, h, x0, loop_idx, return_stats=False):

        """Spider-MoE step with sparse expert execution.

        Router + dynamic edge graph remain fully vectorized. Expert FFNs are
        evaluated only for the top-k selected experts on the tokens that chose
        them (grouped by expert id). Numerically equivalent to the previous
        dense [B,T,E,...] path while avoiding full-expert matmuls.
        """
        # h, x0: [batch, sequence, hidden]
        b, t, width = h.shape
        e = self.num_experts
        m = self.edges_per_expert

        # ----- Router (top-k) -----
        logits = self.router(h) + self.loop_bias[loop_idx].view(1, 1, -1)
        probs = logits.float().softmax(dim=-1)
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ----- Dynamic sparse graph -----
        loop_id = torch.full((b, t), loop_idx, device=h.device, dtype=torch.long)
        loop_emb = self.loop_embedding(loop_id)
        edge_logits = self.edge_generator(torch.cat([h, loop_emb], dim=-1))
        edge_logits = edge_logits.view(b, t, e, e)
        edge_values, edge_indices = edge_logits.topk(m, dim=-1)
        edge_weights = edge_values.float().softmax(dim=-1)  # [B, T, E_src, M]

        # Destination projections: [B, T, E, H]
        # One stacked matmul instead of E separate Linear calls.
        # edge_value[i].weight is [H_out, H_in] with no bias.
        w_edge = torch.stack([layer.weight for layer in self.edge_value], dim=0)  # [E, H, H]
        transformed = torch.einsum("bth,eoh->bteo", h, w_edge)  # [B, T, E, H]


        # Which source experts are active for each token (router top-k).
        # source_active: [B, T, E]
        source_active = h.new_zeros(b, t, e)
        source_active.scatter_(-1, top_indices, 1.0)

        # Gather transformed values at selected destination experts.
        flat_idx = edge_indices.reshape(b, t, e * m)  # [B, T, E*M]
        gather_idx = flat_idx.unsqueeze(-1).expand(b, t, e * m, width)
        gathered = torch.gather(transformed, 2, gather_idx)  # [B, T, E*M, H]
        gathered = gathered.view(b, t, e, m, width)

        # Weight by edge softmax and source activity, then sum over sources & edges.
        w = edge_weights * source_active.unsqueeze(-1)  # [B, T, E, M]
        messages = (gathered * w.unsqueeze(-1)).sum(dim=(2, 3))  # [B, T, H]

        # ----- Sparse Expert FFNs (only top-k selected experts per token) -----
        # Avoids materializing [B, T, E, H_up]. Numerically equivalent to the
        # previous dense path (see sparse_experts.sparse_expert_delta).
        # Offload path: expert chunks stay on CPU; only routed experts hit GPU.
        if self._expert_store is not None:
            delta = self._expert_store.sparse_delta(
                messages, top_indices, top_weights, top_k=self.top_k
            )
        else:
            if self._int4_w_up is not None:
                w_up = self._int4_w_up.dequantize(dtype=messages.dtype)
                w_down = self._int4_w_down.dequantize(dtype=messages.dtype)
                b_up = self._int4_b_up.to(dtype=messages.dtype, device=messages.device)
                b_down = self._int4_b_down.to(dtype=messages.dtype, device=messages.device)
            else:
                w_up = torch.stack([ex[0].weight for ex in self.experts], dim=0)
                b_up = torch.stack([ex[0].bias for ex in self.experts], dim=0)
                w_down = torch.stack([ex[2].weight for ex in self.experts], dim=0)
                b_down = torch.stack([ex[2].bias for ex in self.experts], dim=0)

            delta = sparse_expert_delta(
                messages, top_indices, top_weights,
                w_up, b_up, w_down, b_down,
                num_experts=e, top_k=self.top_k,
                expert_parallel=getattr(self, "expert_parallel", True),
            )

        updated = self.norm(h + self.residual_scale * delta + self.reinject_weight * x0)
        if not return_stats:
            return updated

        load = torch.bincount(top_indices.reshape(-1), minlength=e).float()
        load = load / load.sum().clamp_min(1.0)
        edge_load = torch.bincount(edge_indices.reshape(-1), minlength=e).float()
        edge_load = edge_load / edge_load.sum().clamp_min(1.0)
        edge_importance = edge_weights.new_zeros(e).scatter_add(
            0, edge_indices.reshape(-1), edge_weights.reshape(-1)
        )
        edge_importance = edge_importance / edge_importance.sum().clamp_min(1e-9)
        aux = e * torch.sum(load * probs.mean(dim=(0, 1)))
        edge_aux = e * torch.sum(edge_load * edge_importance)
        confidence = torch.sigmoid(self.exit_head(updated).float()).squeeze(-1)
        stats = {
            "router_probs": probs,
            "expert_load": load,
            "edge_load": edge_load,
            "aux_loss": aux,
            "edge_loss": edge_aux,
            "confidence": confidence,
        }
        return updated, stats


class SpiderWebLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Absolute positions kept for checkpoint compatibility when use_rope=False.
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=cfg.embedding_init_std)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=cfg.embedding_init_std)
        self.prelude = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.num_prelude_layers)])
        self.recurrent = SpiderMoELayer(cfg)
        self.coda = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.num_coda_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # RoPE tables (shared by all attention layers)
        self.use_rope = bool(getattr(cfg, "use_rope", False))
        if self.use_rope:
            head_dim = cfg.hidden_size // cfg.num_heads
            cos, sin = precompute_freqs_cis(
                head_dim, cfg.max_seq_len, theta=getattr(cfg, "rope_theta", 10000.0)
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            for layer in list(self.prelude) + list(self.coda):
                layer.attn.set_rope(self.rope_cos, self.rope_sin)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def _embed(self, input_ids, position=0):
        if input_ids.ndim != 2:
            raise ValueError('input_ids must have shape [batch, sequence]')
        seq_len = input_ids.shape[1]
        if position < 0 or position + seq_len > self.cfg.max_seq_len:
            raise ValueError('input positions exceed cfg.max_seq_len')
        tok = self.token_embedding(input_ids)
        if self.use_rope:
            # Positional information is injected inside attention via RoPE.
            return tok
        positions = torch.arange(position, position + seq_len, device=input_ids.device)
        return tok + self.position_embedding(positions).unsqueeze(0)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        # Keep expert chunks on CPU when offload is enabled (super().to would
        # otherwise pull them onto the compute device).
        rec = getattr(out, "recurrent", None)
        if rec is not None and getattr(rec, "offload_experts", False):
            offload_module_list_to_cpu(rec.experts)
            if rec._expert_store is not None:
                try:
                    rec._expert_store.compute_device = next(out.parameters()).device
                except StopIteration:
                    pass
        return out

    def quantize_moe_experts(self, group_size: int = 64) -> None:
        """Quantize recurrent MoE expert weights to 4-bit (inference)."""
        self.recurrent.quantize_experts(group_size=group_size)

    def _attention_layers(self):
        # Cache only modules that actually perform attention. The recurrent
        # SpiderMoE loop currently has no attention operation.
        return [*self.prelude, *self.coda]

    def allocate_kv_cache(self, batch_size, max_seq_len=None, dtype=None, device=None):
        """Allocate static per-layer K/V buffers for this model's attention layers."""
        max_seq_len = max_seq_len or self.cfg.max_seq_len
        if max_seq_len < 1 or max_seq_len > self.cfg.max_seq_len:
            raise ValueError("max_seq_len must be in [1, cfg.max_seq_len]")
        parameter = next(self.parameters())
        dtype = dtype or parameter.dtype
        device = torch.device(device) if device is not None else parameter.device
        layers = []
        for layer in self._attention_layers():
            shape = (batch_size, layer.attn.heads, max_seq_len, layer.attn.head_dim)
            layers.append(KVLayerCache(
                keys=torch.empty(shape, dtype=dtype, device=device),
                values=torch.empty(shape, dtype=dtype, device=device),
            ))
        return SpiderKVCache(layers=layers, length=0, max_seq_len=max_seq_len)

    def _run_recurrent_full(self, h, x0):
        for loop_idx in range(self.cfg.num_loops):
            h = self.recurrent(h, x0, loop_idx)
        return h

    def _run_recurrent_decode(self, h, x0, early_exit):
        """Decode path with Orbit ActiveSet compaction.

        When early_exit is enabled, rows that pass the exit_head threshold leave
        the orbit and stop consuming further recurrent loops. Compaction uses the
        shared ActiveSet primitive (Orbit 0.2).
        """
        if not early_exit:
            return self._run_recurrent_full(h, x0)
        active = ActiveSet.all_active(h.shape[0], h.device)
        for loop_idx in range(self.cfg.num_loops):
            if active.n_active == 0:
                break
            h_active = compact_active(h, active)
            x0_active = compact_active(x0, active)
            h_new = self.recurrent(h_active, x0_active, loop_idx)
            h = scatter_active(h, active, h_new)
            confidence = torch.sigmoid(self.recurrent.exit_head(h_new).float()).squeeze(-1)
            # exit_head may return [N, 1] or [N]
            if confidence.ndim > 1:
                confidence = confidence[:, 0]
            exited = confidence > self.cfg.early_exit_threshold
            active = active.drop(exited)
        return h

    @torch.no_grad()
    def prefill(self, input_ids, cache=None):
        """Run the prompt once, filling attention K/V buffers; early exit is disabled."""
        self.eval()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("prompt exceeds cfg.max_seq_len")
        if cache is None:
            cache = self.allocate_kv_cache(batch, max_seq_len=self.cfg.max_seq_len)
        if cache.length != 0:
            raise ValueError("prefill expects an empty KV cache")
        if len(cache.layers) != len(self._attention_layers()):
            raise ValueError("KV cache layer count does not match model")

        position = 0
        x0 = self._embed(input_ids, 0)
        h = x0
        layer_index = 0
        for layer in self.prelude:
            h = layer.forward_cached(h, cache.layers[layer_index], position)
            layer_index += 1
        h = self._run_recurrent_full(h, x0)
        for layer in self.coda:
            h = layer.forward_cached(h, cache.layers[layer_index], position)
            layer_index += 1
        cache.length = seq_len
        logits = self.lm_head(self.final_norm(h))
        return logits, cache

    @torch.no_grad()
    def decode_step(self, input_ids, cache: SpiderKVCache, early_exit=False):
        """Process only new token(s), attending to the cached prefix."""
        self.eval()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] < 1:
            raise ValueError("decode_step requires at least one token")
        batch, new_tokens = input_ids.shape
        if cache.length + new_tokens > cache.max_seq_len:
            raise ValueError("KV cache capacity exceeded; rebuild/slide the context first")
        if len(cache.layers) != len(self._attention_layers()):
            raise ValueError("KV cache layer count does not match model")
        position = cache.length
        x0 = self._embed(input_ids, position)
        h = x0
        layer_index = 0
        for layer in self.prelude:
            h = layer.forward_cached(h, cache.layers[layer_index], position)
            layer_index += 1
        if new_tokens == 1:
            h = self._run_recurrent_decode(h, x0, early_exit)
        else:
            # Chunked decode keeps the same full-loop semantics for each token.
            h = self._run_recurrent_full(h, x0)
        for layer in self.coda:
            h = layer.forward_cached(h, cache.layers[layer_index], position)
            layer_index += 1
        cache.length += new_tokens
        logits = self.lm_head(self.final_norm(h))
        return logits, cache

    def allocate_global_paged_cache(
        self, num_pages, page_size=16, max_requests=32,
        max_blocks_per_request=None, dtype=None, device=None,
        fp8_scale_mode: str = "page",
    ):
        """Allocate one global physical page pool shared by requests and layers."""
        parameter = next(self.parameters())
        max_blocks_per_request = (
            max_blocks_per_request
            or (self.cfg.max_seq_len + page_size - 1) // page_size
        )
        return GlobalPagedKVCache(
            num_layers=len(self._attention_layers()),
            num_pages=num_pages,
            page_size=page_size,
            num_heads=self.cfg.num_heads,
            head_dim=self.cfg.hidden_size // self.cfg.num_heads,
            max_requests=max_requests,
            max_blocks_per_request=max_blocks_per_request,
            dtype=dtype or parameter.dtype,
            device=device or parameter.device,
            fp8_scale_mode=fp8_scale_mode,
        )

    def _ensure_paged_requests(self, cache, request_ids, batch_size):
        if request_ids is None:
            request_ids = [f"spider-request-{i}" for i in range(batch_size)]
        if len(request_ids) != batch_size:
            raise ValueError("request_ids length must equal batch size")
        for request_id in request_ids:
            try:
                cache.sequence_length(request_id)
            except KeyError:
                cache.allocate(request_id)
        return request_ids

    @torch.no_grad()
    def prefill_paged(self, input_ids, cache, request_ids=None):
        """Prefill prompt into a global paged cache; all requests must be empty."""
        self.eval()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("prompt exceeds cfg.max_seq_len")
        request_ids = self._ensure_paged_requests(cache, request_ids, batch)
        if any(cache.sequence_length(rid) != 0 for rid in request_ids):
            raise ValueError("prefill_paged requires empty request caches")
        x0 = self._embed(input_ids, 0)
        h = x0
        layer_index = 0
        position = 0
        for layer in self.prelude:
            h = layer.forward_paged(h, cache, request_ids, layer_index, position)
            layer_index += 1
        h = self._run_recurrent_full(h, x0)
        for layer in self.coda:
            h = layer.forward_paged(h, cache, request_ids, layer_index, position)
            layer_index += 1
        logits = self.lm_head(self.final_norm(h))
        return logits, cache

    @torch.no_grad()
    def decode_step_paged(self, input_ids, cache, request_ids, early_exit=False):
        """Decode new token(s) using global physical pages and per-request tables."""
        self.eval()
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, new_tokens>=1]")
        batch, new_tokens = input_ids.shape
        request_ids = self._ensure_paged_requests(cache, request_ids, batch)
        lengths = [cache.sequence_length(rid) for rid in request_ids]
        if len(set(lengths)) != 1:
            raise ValueError("batched paged decode currently requires equal sequence lengths")
        position = lengths[0]
        if position + new_tokens > cache.max_blocks_per_request * cache.page_size:
            raise ValueError("paged cache capacity exceeded")
        x0 = self._embed(input_ids, position)
        h = x0
        layer_index = 0
        for layer in self.prelude:
            h = layer.forward_paged(h, cache, request_ids, layer_index, position)
            layer_index += 1
        if new_tokens == 1:
            h = self._run_recurrent_decode(h, x0, early_exit)
        else:
            h = self._run_recurrent_full(h, x0)
        for layer in self.coda:
            h = layer.forward_paged(h, cache, request_ids, layer_index, position)
            layer_index += 1
        logits = self.lm_head(self.final_norm(h))
        return logits, cache

    @torch.no_grad()
    def generate_paged(
        self, input_ids, cache, request_ids=None, max_new_tokens=100,
        temperature=0.8, top_p=0.9, early_exit=False, eos_token_id=None,
    ):
        """Generate using global paged cache; suitable for shared serving pools."""
        self.eval()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if max_new_tokens <= 0:
            return input_ids
        batch = input_ids.shape[0]
        request_ids = self._ensure_paged_requests(cache, request_ids, batch)
        context = input_ids[:, -self.cfg.max_seq_len:]
        logits, _ = self.prefill_paged(context, cache, request_ids)
        output = input_ids
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(float(temperature), 1e-5)
            if top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                probs = sorted_logits.softmax(-1)
                remove = probs.cumsum(-1) - probs > top_p
                sorted_logits[remove] = -float("inf")
                filtered = torch.full_like(next_logits, -float("inf"))
                filtered.scatter_(1, sorted_idx, sorted_logits)
                next_logits = filtered
            next_id = torch.multinomial(next_logits.softmax(-1), 1)
            output = torch.cat([output, next_id], dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break
            if step + 1 == max_new_tokens:
                break
            if cache.sequence_length(request_ids[0]) >= self.cfg.max_seq_len:
                for rid in request_ids:
                    cache.free(rid)
                context = output[:, -self.cfg.max_seq_len:]
                logits, _ = self.prefill_paged(context, cache, request_ids)
            else:
                logits, _ = self.decode_step_paged(
                    next_id, cache, request_ids, early_exit=early_exit
                )
        return output

    def forward(self, input_ids, targets=None, early_exit=False, return_stats=False):
        x0 = self._embed(input_ids, 0)
        h = x0
        for layer in self.prelude:
            h = layer(h, position=0)
        all_stats = []
        active = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        loops_used = 0
        for loop_idx in range(self.cfg.num_loops):
            if early_exit and not active.any():
                break
            result = self.recurrent(h, x0, loop_idx, return_stats=return_stats)
            if return_stats:
                new_h, stats = result
                all_stats.append(stats)
            else:
                new_h = result
            if early_exit:
                confidence = torch.sigmoid(self.recurrent.exit_head(new_h).float()).squeeze(-1)
                exit_now = confidence > self.cfg.early_exit_threshold
                # Preserve completed token positions while continuing active positions.
                h = torch.where(active.unsqueeze(-1), new_h, h)
                active = active & ~exit_now
            else:
                h = new_h
            loops_used += 1
        for layer in self.coda:
            h = layer(h, position=0)
        logits = self.lm_head(self.final_norm(h))
        output = {"logits": logits, "loops_used": loops_used}
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match input_ids shape {tuple(input_ids.shape)}"
                )
            valid_targets = targets.ne(-100)
            if not valid_targets.any():
                raise ValueError("targets contain no supervised tokens (all labels are -100)")
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,
            )
        if return_stats:
            output["stats"] = all_stats
        return output

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_p=0.9,
                 early_exit=False, eos_token_id=None):
        """Generate using full recurrent depth by default.

        Early exit remains opt-in because exit_head is not trained by the LM
        objective; enabling it by default creates a train/inference mismatch.
        """
        self.eval()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if max_new_tokens <= 0:
            return input_ids
        out = input_ids
        context = out[:, -self.cfg.max_seq_len:]
        cache_capacity = min(
            self.cfg.max_seq_len, context.shape[1] + max_new_tokens
        )
        cache = self.allocate_kv_cache(
            context.shape[0], max_seq_len=cache_capacity
        )
        logits, cache = self.prefill(context, cache)

        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_logits = next_logits / max(float(temperature), 1e-5)
            if top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                probs = sorted_logits.softmax(-1)
                remove = probs.cumsum(-1) - probs > top_p
                sorted_logits[remove] = -float("inf")
                filtered = torch.full_like(next_logits, -float("inf"))
                filtered.scatter_(1, sorted_idx, sorted_logits)
                next_logits = filtered
            next_id = torch.multinomial(next_logits.softmax(-1), num_samples=1)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break
            if step + 1 == max_new_tokens:
                break

            # When the static cache is full, rebuild from the sliding context.
            # Otherwise append only the just-generated token.
            if cache.length >= cache.max_seq_len:
                context = out[:, -self.cfg.max_seq_len:]
                logits, cache = self.prefill(context)
            else:
                logits, cache = self.decode_step(next_id, cache, early_exit=early_exit)
        return out

