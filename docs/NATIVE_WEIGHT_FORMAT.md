# Native weight format — `.spiderw`

Torch-free inference load format for Orbit native backend.

## File layout (little-endian)

```
Offset  Size     Field
0       8        magic = "SPIDERW1"
8       4        version = 1 (uint32)
12      4        flags — bit0=fp16_weights bit1=has_csr bit2=tied_emb
16      4        vocab_size
20      4        hidden_size
24      4        num_heads
28      4        num_experts
32      4        top_k
36      4        edges_per_expert
40      4        num_loops
44      4        expert_ffn_size
48      4        num_prelude_layers
52      4        num_coda_layers
56      4        reserved0
60      4        reserved1
64      8        header_crc64 (0 if unused)
72      8        data_nbytes
80      ...      payload
```

Header size **80 bytes**.

## Section mini-header

```
uint32 section_id
uint32 dtype      // 0=f32, 1=f16, 2=bf16, 3=i32, 4=i8
uint64 nbytes
uint32 rank
uint32 dims[4]
bytes  data
```

## Section IDs

| ID | Name |
|----|------|
| 1 | token_emb |
| 2 | lm_head |
| 30 | router |
| 31 | loop_bias |
| 32 | loop_embedding |
| 33 | edge_generator |
| 34 | edge_value |
| 35 | expert_up [E,H,U] |
| 36 | expert_down [E,U,H] |
| 37 | moe_norm |
| 38 | exit_head |
| 40 | csr_row_ptr |
| 41 | csr_col_idx |
| 42 | csr_edge_w |

See `spider_infer/orbit/weight_loader.py` and `scripts/export_native_weights.py`.
