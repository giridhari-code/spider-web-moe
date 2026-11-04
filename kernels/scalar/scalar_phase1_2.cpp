/* Phase 1.2: dynamic per-token edges (PyTorch edge_generator path). */
#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" int spider_edge_project_f32(const float*, int, int, int, const float*, float*);

extern "C" int spider_edge_topk_f32(
    const float* edge_logits, int B, int E, int M,
    int32_t* edge_indices, float* edge_weights) {
    if (!edge_logits || !edge_indices || !edge_weights || B <= 0 || E <= 0 || M <= 0 || M > E)
        return 1;
    for (int b = 0; b < B; ++b) {
        for (int src = 0; src < E; ++src) {
            const float* row = edge_logits + (static_cast<size_t>(b) * E + src) * E;
            for (int m = 0; m < M; ++m) {
                int best = -1;
                float bestv = -std::numeric_limits<float>::infinity();
                for (int dst = 0; dst < E; ++dst) {
                    bool used = false;
                    for (int q = 0; q < m; ++q)
                        used |= edge_indices[(static_cast<size_t>(b) * E + src) * M + q] == dst;
                    if (used) continue;
                    if (row[dst] > bestv) { bestv = row[dst]; best = dst; }
                }
                edge_indices[(static_cast<size_t>(b) * E + src) * M + m] = best;
                edge_weights[(static_cast<size_t>(b) * E + src) * M + m] = bestv;
            }
            float mx = -std::numeric_limits<float>::infinity();
            for (int m = 0; m < M; ++m)
                mx = std::max(mx, edge_weights[(static_cast<size_t>(b) * E + src) * M + m]);
            float sum = 0.f;
            for (int m = 0; m < M; ++m) {
                float& w = edge_weights[(static_cast<size_t>(b) * E + src) * M + m];
                w = std::exp(w - mx);
                sum += w;
            }
            for (int m = 0; m < M; ++m)
                edge_weights[(static_cast<size_t>(b) * E + src) * M + m] /=
                    std::max(sum, 1e-20f);
        }
    }
    return 0;
}

extern "C" int spider_graph_message_passing_dynamic_f32(
    const float* projected, int B, int H, int E, int K, int M,
    const int32_t* edge_indices, const float* edge_weights,
    const int32_t* assign, float* messages) {
    if (!projected || !edge_indices || !edge_weights || !assign || !messages ||
        B <= 0 || H <= 0 || E <= 0 || K <= 0 || M <= 0)
        return 1;
    std::fill(messages, messages + static_cast<size_t>(B) * H, 0.f);
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            int src = assign[b * K + k];
            if (src < 0) continue;
            if (src >= E) return 2;
            for (int m = 0; m < M; ++m) {
                int dst = edge_indices[(static_cast<size_t>(b) * E + src) * M + m];
                if (dst < 0 || dst >= E) return 2;
                float scale = edge_weights[(static_cast<size_t>(b) * E + src) * M + m];
                const float* p = projected + (static_cast<size_t>(b) * E + dst) * H;
                for (int j = 0; j < H; ++j) messages[b * H + j] += scale * p[j];
            }
        }
    }
    return 0;
}

extern "C" int spider_graph_dynamic_with_edge_value_f32(
    const float* hidden, int B, int H, int E, int K, int M,
    const float* W_edge,
    const int32_t* edge_indices, const float* edge_weights,
    const int32_t* assign, float* messages, float* tmp_projected) {
    if (!hidden || !W_edge || !edge_indices || !edge_weights || !assign || !messages ||
        !tmp_projected)
        return 1;
    int rc = spider_edge_project_f32(hidden, B, H, E, W_edge, tmp_projected);
    if (rc != 0) return rc;
    return spider_graph_message_passing_dynamic_f32(
        tmp_projected, B, H, E, K, M, edge_indices, edge_weights, assign, messages);
}
