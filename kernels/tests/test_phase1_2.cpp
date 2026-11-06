#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_edge_topk() {
    const int B = 1, E = 3, M = 2;
    float logits[B * E * E] = {
        0.f, 1.f, 2.f,
        3.f, 0.f, 1.f,
        0.f, 0.f, 1.f,
    };
    int32_t idx[B * E * M];
    float w[B * E * M];
    assert(spider_edge_topk_f32(logits, B, E, M, idx, w) == 0);
    assert(idx[0] == 2 && idx[1] == 1);
    assert(std::fabs(w[0] + w[1] - 1.f) < 1e-5f);
}

static void test_dynamic_graph() {
    const int B = 1, H = 2, E = 2, K = 1, M = 1;
    float projected[B * E * H] = {1.f, 0.f, 0.f, 4.f};
    int32_t edge_idx[B * E * M] = {1, 0};
    float edge_w[B * E * M] = {0.5f, 1.f};
    int32_t assign[K] = {0};
    float messages[H];
    assert(spider_graph_message_passing_dynamic_f32(
               projected, B, H, E, K, M, edge_idx, edge_w, assign, messages) == 0);
    assert(std::fabs(messages[0] - 0.f) < 1e-5f);
    assert(std::fabs(messages[1] - 2.f) < 1e-5f);
}

static void test_dynamic_with_edge_value() {
    const int B = 1, H = 2, E = 2, K = 1, M = 1;
    float h[H] = {1.f, 0.f};
    float W[E * H * H] = {0};
    W[0] = 1.f; W[3] = 1.f; W[4] = 1.f; W[7] = 1.f;
    int32_t edge_idx[E * M] = {1, 0};
    float edge_w[E * M] = {1.f, 1.f};
    int32_t assign[K] = {0};
    float messages[H], tmp[B * E * H];
    assert(spider_graph_dynamic_with_edge_value_f32(
               h, B, H, E, K, M, W, edge_idx, edge_w, assign, messages, tmp) == 0);
    for (int j = 0; j < H; ++j) assert(std::isfinite(messages[j]));
}

int main() {
    test_edge_topk();
    test_dynamic_graph();
    test_dynamic_with_edge_value();
    std::printf("phase1.2 tests OK backend=%s\n", spider_backend_name());
    return 0;
}
