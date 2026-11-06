# Phase 2.4 — CUDA causal attention + KV

## API

```c
size_t spider_cuda_attention_kv_workspace_bytes(int B, int T, int H);

int spider_cuda_attention_causal_kv_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    float* keys, float* values, int max_T, int position, int* length_inout,
    float* out, float* workspace, void* stream);
```

Pipeline: QKV proj → write KV → causal softmax·V → out proj.

Notes (v0): FP32; RoPE on host/Phase 2.4.1; tiling/FlashAttention later.

Build: `-DSPIDER_ENABLE_CUDA=ON`
