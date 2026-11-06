# Phase 2.1.1 — RoPE

## API

```c
int spider_rope_precompute_f32(int head_dim, int max_T, float theta,
                               float* cos_out, float* sin_out);
/* cos/sin: [max_T, head_dim/2] */

int spider_rope_apply_f32(float* x, int B, int n_heads, int T, int head_dim,
                          const float* cos_table, const float* sin_table,
                          int position);
/* x: [B, n_heads, T, head_dim] in-place; NeoX pair (i, i+half) */
```

Default theta=10000. Apply on Q and K after projection, before attention scores.

Test: `spider_kernel_phase2_rope`
