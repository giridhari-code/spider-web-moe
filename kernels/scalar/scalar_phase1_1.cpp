/* Phase 1.1: edge_value projection, graph v2, full MoE loop compose (scalar). */
#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" int spider_scalar_router_f32(const float*, const float*, int, int, int, int, float*, int32_t*);
extern "C" int spider_scalar_expert_grouped_gemm_f32(const float*, int, int, int, int, int,
                                                    const int32_t*, const float*, const float*,
                                                    const float*, float*);
extern "C" int spider_scalar_loop_epilogue_f32(const float*, const float*, const float*, int, int,
                                              float, float, const float*, const float*, float, float*);
extern "C" int spider_scalar_early_exit_f32(const float*, const float*, int, int, float, uint8_t*);

extern "C" int spider_edge_project_f32(
    const float* hidden, int B, int H, int E, const float* W_edge, float* out_proj) {
    if (!hidden || !W_edge || !out_proj || B <= 0 || H <= 0 || E <= 0) return 1;
    for (int b = 0; b < B; ++b) {
        for (int e = 0; e < E; ++e) {
            const float* W = W_edge + (static_cast<size_t>(e) * H * H);
            for (int j = 0; j < H; ++j) {
                double acc = 0.0;
                const float* row = W + static_cast<size_t>(j) * H;
                for (int i = 0; i < H; ++i) acc += static_cast<double>(hidden[b * H + i]) * row[i];
                out_proj[(static_cast<size_t>(b) * E + e) * H + j] = static_cast<float>(acc);
            }
        }
    }
    return 0;
}

extern "C" int spider_graph_message_passing_v2_f32(
    const float* projected, int B, int H, int E, int K,
    const int32_t* row_ptr, const int32_t* col_idx,
    const float* edge_weights, int nnz, int loop_idx,
    const int32_t* assign, float* messages) {
    if (!projected || !row_ptr || !col_idx || !edge_weights || !assign || !messages ||
        B <= 0 || H <= 0 || E <= 0 || K <= 0 || nnz < 0 || loop_idx < 0 ||
        row_ptr[0] != 0 || row_ptr[E] != nnz)
        return 1;
    std::fill(messages, messages + static_cast<size_t>(B) * H, 0.f);
    const float* ew = edge_weights + static_cast<size_t>(loop_idx) * nnz;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            int src = assign[b * K + k];
            if (src < 0) continue;
            if (src >= E) return 2;
            for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
                int dst = col_idx[edge];
                if (dst < 0 || dst >= E) return 2;
                float scale = ew[edge];
                const float* p = projected + (static_cast<size_t>(b) * E + dst) * H;
                for (int j = 0; j < H; ++j) messages[b * H + j] += scale * p[j];
            }
        }
    }
    return 0;
}

extern "C" int spider_moe_loop_step_f32(
    const float* h, const float* x0, int B, int H, int E, int K, int U, int loop_idx,
    const float* router_w, const float* loop_bias,
    const int32_t* row_ptr, const int32_t* col_idx, const float* edge_w, int nnz,
    const float* W_edge, const float* W_up, const float* W_down,
    const float* ln_weight, const float* ln_bias, float alpha, float beta, float eps,
    const float* exit_proj, float exit_threshold,
    float* h_next, uint8_t* exit_mask,
    int32_t* tmp_assign, float* tmp_route, float* tmp_messages) {
    if (!h || !x0 || !router_w || !loop_bias || !W_up || !W_down || !ln_weight || !ln_bias ||
        !h_next || !tmp_assign || !tmp_route || !tmp_messages || B <= 0 || H <= 0 || E <= 0 ||
        K <= 0 || U <= 0)
        return 1;

    std::vector<float> logits(static_cast<size_t>(B) * E);
    for (int b = 0; b < B; ++b) {
        for (int e = 0; e < E; ++e) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i)
                acc += static_cast<double>(h[b * H + i]) * router_w[i * E + e];
            logits[b * E + e] = static_cast<float>(acc);
        }
    }

    int rc = spider_scalar_router_f32(logits.data(), loop_bias, B, E, K, loop_idx, tmp_route,
                                     tmp_assign);
    if (rc != 0) return rc;

    std::vector<float> projected;
    const float* msg_src = h;
    if (W_edge && row_ptr && col_idx && edge_w && nnz >= 0) {
        projected.resize(static_cast<size_t>(B) * E * H);
        rc = spider_edge_project_f32(h, B, H, E, W_edge, projected.data());
        if (rc != 0) return rc;
        rc = spider_graph_message_passing_v2_f32(projected.data(), B, H, E, K, row_ptr, col_idx,
                                                edge_w, nnz, loop_idx, tmp_assign, tmp_messages);
        if (rc != 0) return rc;
        msg_src = tmp_messages;
    } else {
        std::memcpy(tmp_messages, h, sizeof(float) * static_cast<size_t>(B) * H);
        msg_src = tmp_messages;
    }

    std::vector<float> delta(static_cast<size_t>(B) * H);
    rc = spider_scalar_expert_grouped_gemm_f32(msg_src, B, H, U, E, K, tmp_assign, tmp_route, W_up,
                                              W_down, delta.data());
    if (rc != 0) return rc;

    rc = spider_scalar_loop_epilogue_f32(h, delta.data(), x0, B, H, alpha, beta, ln_weight, ln_bias,
                                        eps, h_next);
    if (rc != 0) return rc;

    if (exit_mask && exit_proj) {
        rc = spider_scalar_early_exit_f32(h_next, exit_proj, B, H, exit_threshold, exit_mask);
        if (rc != 0) return rc;
    } else if (exit_mask) {
        std::memset(exit_mask, 0, static_cast<size_t>(B));
    }
    return 0;
}
