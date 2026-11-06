#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_rope_precompute_and_apply() {
    const int D = 8, max_T = 16, B = 1, nh = 1, T = 2;
    std::vector<float> cos(max_T * (D / 2)), sin(max_T * (D / 2));
    assert(spider_rope_precompute_f32(D, max_T, 10000.f, cos.data(), sin.data()) == 0);
    for (float v : cos) assert(std::isfinite(v));
    for (float v : sin) assert(std::isfinite(v));

    std::vector<float> x(B * nh * T * D, 0.f);
    x[0] = 1.f;
    x[D / 2] = 0.f;
    assert(spider_rope_apply_f32(x.data(), B, nh, T, D, cos.data(), sin.data(), 0) == 0);
    for (float v : x) assert(std::isfinite(v));
    assert(std::fabs(x[0] - 1.f) < 0.05f);
}

int main() {
    test_rope_precompute_and_apply();
    std::printf("phase2.1.1 RoPE OK backend=%s\n", spider_backend_name());
    return 0;
}
