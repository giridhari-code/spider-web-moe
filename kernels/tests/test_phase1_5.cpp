#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void ref_expert(const float* x, int B, int H, int U, int E, int K,
                       const int32_t* ids, const float* rw,
                       const float* up, const float* down, float* out) {
    for (int i = 0; i < B * H; ++i) out[i] = 0.f;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            int e = ids[b * K + k];
            if (e < 0 || e >= E) continue;
            float r = rw[b * K + k];
            for (int u = 0; u < U; ++u) {
                float a = 0.f;
                for (int i = 0; i < H; ++i)
                    a += x[b * H + i] * up[((e * H + i) * U) + u];
                float c = 0.7978845608f * (a + 0.044715f * a * a * a);
                float g = 0.5f * a * (1.f + std::tanh(c));
                for (int j = 0; j < H; ++j)
                    out[b * H + j] += r * g * down[((e * U + u) * H) + j];
            }
        }
    }
}

static void test_scalar_matches_ref() {
    const int B = 2, H = 8, U = 16, E = 4, K = 2;
    std::vector<float> x(B * H), up(E * H * U), down(E * U * H), rw(B * K);
    std::vector<int32_t> ids(B * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float)(i % 7);
    for (size_t i = 0; i < up.size(); ++i) up[i] = 0.02f * (float)((i % 5) - 2);
    for (size_t i = 0; i < down.size(); ++i) down[i] = 0.015f * (float)((i % 3) - 1);
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < K; ++k) {
            ids[b * K + k] = (b + k) % E;
            rw[b * K + k] = 1.f / K;
        }
    std::vector<float> out(B * H), ref(B * H);
    assert(spider_expert_grouped_gemm_f32(x.data(), B, H, U, E, K, ids.data(), rw.data(),
                                         up.data(), down.data(), out.data()) == 0);
    ref_expert(x.data(), B, H, U, E, K, ids.data(), rw.data(), up.data(), down.data(), ref.data());
    float max_err = 0.f;
    for (int i = 0; i < B * H; ++i)
        max_err = std::max(max_err, std::fabs(out[i] - ref[i]));
    assert(max_err < 1e-4f);
}

int main() {
    test_scalar_matches_ref();
    std::printf("phase1.5 scalar expert parity OK backend=%s\n", spider_backend_name());
    return 0;
}
