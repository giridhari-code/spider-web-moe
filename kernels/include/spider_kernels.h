#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #define SPIDER_API __declspec(dllexport)
#else
  #define SPIDER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpiderKVCache {
    float* keys;
    float* values;
    int B;
    int n_heads;
    int max_T;
    int head_dim;
    int length;
} SpiderKVCache;

SPIDER_API int spider_router_f32(
    const float* logits, const float* loop_bias, int B, int E, int K, int loop_idx,
    float* out_weights, int32_t* out_indices);

SPIDER_API int spider_router_f16(
    const uint16_t* logits, const uint16_t* loop_bias, int B, int E, int K, int loop_idx,
    float* out_weights, int32_t* out_indices);

SPIDER_API int spider_graph_message_passing_f32(
    const float* hidden, int B, int H, int E, int K,
    const int32_t* adj_row_ptr, const int32_t* adj_col_idx,
    const float* edge_weights, int nnz, int loop_idx,
    const int32_t* expert_assign, const float* route_weights,
    float* messages);

SPIDER_API int spider_graph_message_passing_f16(
    const uint16_t* hidden, int B, int H, int E, int K,
    const int32_t* adj_row_ptr, const int32_t* adj_col_idx,
    const float* edge_weights, int nnz, int loop_idx,
    const int32_t* expert_assign, const float* route_weights,
    float* messages);

SPIDER_API int spider_edge_project_f32(
    const float* hidden, int B, int H, int E,
    const float* W_edge, float* out_proj);

SPIDER_API int spider_graph_message_passing_v2_f32(
    const float* projected, int B, int H, int E, int K,
    const int32_t* row_ptr, const int32_t* col_idx,
    const float* edge_weights, int nnz, int loop_idx,
    const int32_t* assign, float* messages);

SPIDER_API int spider_edge_topk_f32(
    const float* edge_logits, int B, int E, int M,
    int32_t* edge_indices, float* edge_weights);

SPIDER_API int spider_graph_message_passing_dynamic_f32(
    const float* projected, int B, int H, int E, int K, int M,
    const int32_t* edge_indices, const float* edge_weights,
    const int32_t* assign, float* messages);

SPIDER_API int spider_graph_dynamic_with_edge_value_f32(
    const float* hidden, int B, int H, int E, int K, int M,
    const float* W_edge,
    const int32_t* edge_indices, const float* edge_weights,
    const int32_t* assign, float* messages, float* tmp_projected);

SPIDER_API int spider_edge_generator_f32(
    const float* hidden, int B, int H, int E,
    const float* loop_emb, int loop_emb_dim,
    const float* W_gen, float* out_logits);

SPIDER_API int spider_edge_generator_topk_f32(
    const float* hidden, int B, int H, int E, int M,
    const float* loop_emb, int loop_emb_dim, const float* W_gen,
    int32_t* edge_indices, float* edge_weights, float* tmp_logits);

SPIDER_API int spider_graph_from_generator_f32(
    const float* hidden, int B, int H, int E, int K, int M,
    const float* loop_emb, int loop_emb_dim, const float* W_gen, const float* W_edge,
    const int32_t* assign, float* messages,
    float* tmp_logits, int32_t* tmp_edge_idx, float* tmp_edge_w, float* tmp_projected);

SPIDER_API int spider_expert_grouped_gemm_f32(
    const float* tokens, int B, int H, int H_up, int E, int K,
    const int32_t* expert_assign, const float* route_weights,
    const float* W_up, const float* W_down, float* output);

SPIDER_API int spider_expert_grouped_gemm_f16(
    const float* tokens, int B, int H, int H_up, int E, int K,
    const int32_t* expert_assign, const float* route_weights,
    const uint16_t* W_up, const uint16_t* W_down, float* output);

SPIDER_API int spider_loop_epilogue_f32(
    const float* h, const float* delta, const float* input_embed,
    int B, int H, float alpha, float beta,
    const float* ln_weight, const float* ln_bias, float eps, float* h_next);

SPIDER_API int spider_loop_epilogue_f16(
    const float* h, const float* delta, const float* input_embed,
    int B, int H, float alpha, float beta,
    const float* ln_weight, const float* ln_bias, float eps, float* h_next);

SPIDER_API int spider_early_exit_f32(
    const float* h, const float* exit_proj, int B, int H,
    float threshold, uint8_t* should_exit);

SPIDER_API int spider_moe_loop_step_f32(
    const float* h, const float* x0, int B, int H, int E, int K, int U, int loop_idx,
    const float* router_w, const float* loop_bias,
    const int32_t* row_ptr, const int32_t* col_idx, const float* edge_w, int nnz,
    const float* W_edge, const float* W_up, const float* W_down,
    const float* ln_weight, const float* ln_bias, float alpha, float beta, float eps,
    const float* exit_proj, float exit_threshold,
    float* h_next, uint8_t* exit_mask,
    int32_t* tmp_assign, float* tmp_route, float* tmp_messages);

SPIDER_API size_t spider_attention_workspace_bytes(int B, int T, int H, int n_heads);

SPIDER_API int spider_attention_causal_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    float* out, float* workspace);

SPIDER_API size_t spider_kv_bytes(int B, int n_heads, int max_T, int head_dim);
SPIDER_API int spider_kv_init(SpiderKVCache* cache, float* keys, float* values,
                              int B, int n_heads, int max_T, int head_dim);
SPIDER_API int spider_kv_reset(SpiderKVCache* cache);
SPIDER_API size_t spider_attention_kv_workspace_bytes(int B, int T, int H);
SPIDER_API int spider_attention_causal_kv_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    SpiderKVCache* cache, int position,
    float* out, float* workspace);

/* Phase 2.1.1 RoPE */
SPIDER_API int spider_rope_precompute_f32(
    int head_dim, int max_T, float theta, float* cos_out, float* sin_out);
SPIDER_API int spider_rope_apply_f32(
    float* x, int B, int n_heads, int T, int head_dim,
    const float* cos_table, const float* sin_table, int position);

SPIDER_API const char* spider_backend_name(void);
SPIDER_API uint32_t spider_backend_caps(void);

#ifdef __cplusplus
}
#endif
