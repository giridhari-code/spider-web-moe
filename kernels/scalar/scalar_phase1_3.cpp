/* Phase 1.3: edge_generator GEMM + fused edge logits→topk. */
#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" int spider_edge_topk_f32(const float*, int, int, int, int32_t*, float*);
extern "C" int spider_edge_project_f32(const float*, int, int, int, const float*, float*);
extern "C" int spider_graph_message_passing_dynamic_f32(
    const float*, int, int, int, int, int, const int32_t*, const float*, const int32_t*, float*);

extern "C" int spider_edge_generator_f32(
    const float* hidden, int B, int H, int E,
    const float* loop_emb, int loop_emb_dim,
    const float* W_gen, float* out_logits) {
    if (!hidden || !loop_emb || !W_gen || !out_logits || B <= 0 || H <= 0 || E <= 0 ||
        loop_emb_dim < 0)
        return 1;
    const int in_dim = H + loop_emb_dim;
    const int out_dim = E * E;
    std::vector<float> feat(static_cast<size_t>(B) * in_dim);
    for (int b = 0; b < B; ++b) {
        std::memcpy(feat.data() + static_cast<size_t>(b) * in_dim, hidden + b * H,
                    sizeof(float) * static_cast<size_t>(H));
        if (loop_emb_dim > 0) {
            std::memcpy(feat.data() + static_cast<size_t>(b) * in_dim + H, loop_emb,
                        sizeof(float) * static_cast<size_t>(loop_emb_dim));
        }
    }
    for (int b = 0; b < B; ++b) {
        const float* x = feat.data() + static_cast<size_t>(b) * in_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* row = W_gen + static_cast<size_t>(o) * in_dim;
            double acc = 0.0;
            for (int i = 0; i < in_dim; ++i) acc += static_cast<double>(x[i]) * row[i];
            out_logits[static_cast<size_t>(b) * out_dim + o] = static_cast<float>(acc);
        }
    }
    return 0;
}

extern "C" int spider_edge_generator_topk_f32(
    const float* hidden, int B, int H, int E, int M,
    const float* loop_emb, int loop_emb_dim, const float* W_gen,
    int32_t* edge_indices, float* edge_weights, float* tmp_logits) {
    if (!tmp_logits) return 1;
    int rc = spider_edge_generator_f32(hidden, B, H, E, loop_emb, loop_emb_dim, W_gen, tmp_logits);
    if (rc != 0) return rc;
    return spider_edge_topk_f32(tmp_logits, B, E, M, edge_indices, edge_weights);
}

extern "C" int spider_graph_from_generator_f32(
    const float* hidden, int B, int H, int E, int K, int M,
    const float* loop_emb, int loop_emb_dim, const float* W_gen, const float* W_edge,
    const int32_t* assign,
    float* messages,
    float* tmp_logits, int32_t* tmp_edge_idx, float* tmp_edge_w, float* tmp_projected) {
    if (!hidden || !W_gen || !W_edge || !assign || !messages || !tmp_logits || !tmp_edge_idx ||
        !tmp_edge_w || !tmp_projected)
        return 1;
    int rc = spider_edge_generator_topk_f32(hidden, B, H, E, M, loop_emb, loop_emb_dim, W_gen,
                                           tmp_edge_idx, tmp_edge_w, tmp_logits);
    if (rc != 0) return rc;
    rc = spider_edge_project_f32(hidden, B, H, E, W_edge, tmp_projected);
    if (rc != 0) return rc;
    return spider_graph_message_passing_dynamic_f32(tmp_projected, B, H, E, K, M, tmp_edge_idx,
                                                    tmp_edge_w, assign, messages);
}
