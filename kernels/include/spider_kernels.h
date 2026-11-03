#pragma once
#include <stdbool.h>
#include <stdint.h>

#if defined(_WIN32)
  #define SPIDER_API __declspec(dllexport)
#else
  #define SPIDER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Phase-1 native kernels use contiguous FP32 tensors. See docs/KERNEL_BACKENDS.md.
// Return 0 on success, nonzero on invalid arguments.
SPIDER_API int spider_router_f32(
    const float* logits, const float* loop_bias, int B, int E, int K, int loop_idx,
    float* out_weights, int32_t* out_indices);

SPIDER_API int spider_graph_message_passing_f32(
    const float* hidden, int B, int H, int E, int K,
    const int32_t* adj_row_ptr, const int32_t* adj_col_idx,
    const float* edge_weights, int nnz, int loop_idx,
    const int32_t* expert_assign, const float* route_weights,
    float* messages);

SPIDER_API int spider_expert_grouped_gemm_f32(
    const float* tokens, int B, int H, int H_up, int E, int K,
    const int32_t* expert_assign, const float* route_weights,
    const float* W_up, const float* W_down, float* output);

SPIDER_API int spider_loop_epilogue_f32(
    const float* h, const float* delta, const float* input_embed,
    int B, int H, float alpha, float beta,
    const float* ln_weight, const float* ln_bias, float eps, float* h_next);

SPIDER_API int spider_early_exit_f32(
    const float* h, const float* exit_proj, int B, int H,
    float threshold, uint8_t* should_exit);

// Explicit backend availability; ARM function selects NEON when compiled for ARM.
SPIDER_API const char* spider_backend_name(void);

#ifdef __cplusplus
}
#endif
