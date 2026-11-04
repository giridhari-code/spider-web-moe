#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

int spider_cuda_router_f32(
    const float*, const float*, int, int, int, int, float*, int32_t*, void* stream);

int spider_cuda_graph_message_passing_f32(
    const float*, int, int, int, int,
    const int32_t*, const int32_t*, const float*, int, int,
    const int32_t*, const float*, float*, void* stream);

int spider_cuda_expert_grouped_gemm_f32(
    const float*, int, int, int, int, int,
    const int32_t*, const float*, const float*, const float*, float*, void* stream);

/* Phase 1.5 tiled path */
size_t spider_cuda_expert_workspace_bytes(int B, int K, int U);

int spider_cuda_expert_grouped_gemm_v2_f32(
    const float* tokens, int B, int H, int U, int E, int K,
    const int32_t* assign, const float* route,
    const float* W_up, const float* W_down,
    float* output, float* workspace, void* stream);

int spider_cuda_loop_epilogue_f32(
    const float*, const float*, const float*, int, int, float, float,
    const float*, const float*, float, float*, void* stream);

int spider_cuda_early_exit_f32(
    const float*, const float*, int, int, float, uint8_t*, void* stream);

int spider_cuda_moe_loop_step_f32(
    const float* h, const float* x0, int B, int H, int E, int K, int U, int loop_idx,
    const float* router_w, const float* loop_bias,
    const int32_t* row_ptr, const int32_t* col_idx, const float* edge_w, int nnz,
    const float* W_up, const float* W_down,
    const float* ln_weight, const float* ln_bias, float alpha, float beta, float eps,
    const float* exit_proj, float exit_threshold,
    float* h_next, uint8_t* exit_mask,
    int32_t* tmp_assign, float* tmp_route, float* tmp_messages,
    void* stream);

#ifdef __cplusplus
}
#endif
