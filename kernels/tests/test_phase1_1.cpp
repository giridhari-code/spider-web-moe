#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

static void test_edge_project_identity() {
    const int B = 2, H = 4, E = 3;
    std::vector<float> h(B * H), W(E * H * H, 0.f), out(B * E * H);
    for (int i = 0; i < B * H; ++i) h[i] = 0.1f * (i + 1);
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < H; ++j) W[(e * H + j) * H + j] = 1.f;
    assert(spider_edge_project_f32(h.data(), B, H, E, W.data(), out.data()) == 0);
    for (int b = 0; b < B; ++b)
        for (int e = 0; e < E; ++e)
            for (int j = 0; j < H; ++j)
                assert(std::fabs(out[(b * E + e) * H + j] - h[b * H + j]) < 1e-5f);
}

static void test_graph_v2_dest_proj() {
    const int B = 1, H = 2, E = 2, K = 1, nnz = 1;
    float projected[B * E * H] = {1.f, 0.f, 0.f, 2.f};
    int32_t row_ptr[3] = {0, 1, 1};
    int32_t col_idx[1] = {1};
    float edge_w[1] = {0.5f};
    int32_t assign[K] = {0};
    float messages[H];
    assert(spider_graph_message_passing_v2_f32(projected, B, H, E, K, row_ptr, col_idx, edge_w, nnz,
                                              0, assign, messages) == 0);
    assert(std::fabs(messages[0] - 0.f) < 1e-6f);
    assert(std::fabs(messages[1] - 1.f) < 1e-6f);
}

static void test_moe_loop_step_finite() {
    const int B = 1, H = 4, E = 2, K = 1, U = 4;
    std::vector<float> h(B * H, 0.05f), x0(B * H, 0.01f);
    std::vector<float> router(H * E, 0.02f), bias(E, 0.f);
    std::vector<float> W_edge(E * H * H, 0.f), W_up(E * H * U, 0.01f), W_down(E * U * H, 0.01f);
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < H; ++j) W_edge[(e * H + j) * H + j] = 1.f;
    std::vector<float> gamma(H, 1.f), lb(H, 0.f), exit_p(H, 0.1f);
    int32_t row_ptr[3] = {0, 1, 1};
    int32_t col_idx[1] = {0};
    float edge_w[1] = {1.f};
    std::vector<float> h_next(B * H), route(B * K), messages(B * H);
    std::vector<int32_t> assign(B * K);
    uint8_t mask[B];
    int rc = spider_moe_loop_step_f32(
        h.data(), x0.data(), B, H, E, K, U, 0, router.data(), bias.data(), row_ptr, col_idx, edge_w,
        1, W_edge.data(), W_up.data(), W_down.data(), gamma.data(), lb.data(), 1.f, 0.f, 1e-5f,
        exit_p.data(), 0.99f, h_next.data(), mask, assign.data(), route.data(), messages.data());
    assert(rc == 0);
    for (float v : h_next) assert(std::isfinite(v));
}

int main() {
    test_edge_project_identity();
    test_graph_v2_dest_proj();
    test_moe_loop_step_finite();
    std::printf("phase1.1 tests OK backend=%s\n", spider_backend_name());
    return 0;
}
