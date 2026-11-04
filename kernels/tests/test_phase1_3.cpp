#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

static void test_edge_generator() {
    const int B = 1, H = 2, E = 2, D = 1;
    float h[H] = {1.f, 0.f};
    float emb[D] = {0.f};
    float W[(E * E) * (H + D)];
    std::memset(W, 0, sizeof(W));
    W[1 * 3 + 0] = 5.f;
    W[2 * 3 + 0] = 5.f;
    float logits[E * E];
    assert(spider_edge_generator_f32(h, B, H, E, emb, D, W, logits) == 0);
    assert(logits[1] > logits[0]);
}

static void test_generator_topk() {
    const int B = 1, H = 2, E = 2, M = 1, D = 0;
    float h[H] = {1.f, 0.f};
    float emb[1] = {0.f};
    float W[(E * E) * H];
    std::memset(W, 0, sizeof(W));
    W[1 * H + 0] = 3.f;
    W[2 * H + 0] = 3.f;
    int32_t idx[E * M];
    float wt[E * M];
    float tmp[E * E];
    assert(spider_edge_generator_topk_f32(h, B, H, E, M, emb, D, W, idx, wt, tmp) == 0);
    assert(idx[0] == 1);
    assert(std::fabs(wt[0] - 1.f) < 1e-5f);
}

static void test_graph_from_generator() {
    const int B = 1, H = 2, E = 2, K = 1, M = 1, D = 0;
    float h[H] = {1.f, 0.f};
    float emb[1] = {0.f};
    float W_gen[(E * E) * H];
    std::memset(W_gen, 0, sizeof(W_gen));
    W_gen[1 * H + 0] = 2.f;
    float W_edge[E * H * H];
    std::memset(W_edge, 0, sizeof(W_edge));
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < H; ++j) W_edge[(e * H + j) * H + j] = 1.f;
    int32_t assign[K] = {0};
    float messages[H], logits[E * E], ew[E * M], proj[B * E * H];
    int32_t eidx[E * M];
    assert(spider_graph_from_generator_f32(h, B, H, E, K, M, emb, D, W_gen, W_edge, assign,
                                          messages, logits, eidx, ew, proj) == 0);
    for (int j = 0; j < H; ++j) assert(std::isfinite(messages[j]));
}

int main() {
    test_edge_generator();
    test_generator_topk();
    test_graph_from_generator();
    std::printf("phase1.3 tests OK backend=%s\n", spider_backend_name());
    return 0;
}
