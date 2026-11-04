#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_attention_shapes() {
    const int B = 1, T = 4, H = 8, nh = 2;
    std::vector<float> x(B * T * H, 0.1f), Wq(3 * H * H, 0.f), Wo(H * H, 0.f);
    for (int i = 0; i < H; ++i) {
        Wq[i * H + i] = 1.f;
        Wq[(H + i) * H + i] = 1.f;
        Wq[(2 * H + i) * H + i] = 1.f;
        Wo[i * H + i] = 1.f;
    }
    size_t ws = spider_attention_workspace_bytes(B, T, H, nh);
    std::vector<float> workspace(ws / sizeof(float) + 8);
    std::vector<float> out(B * T * H);
    assert(spider_attention_causal_f32(x.data(), B, T, H, nh, Wq.data(), Wo.data(), out.data(),
                                      workspace.data()) == 0);
    for (float v : out) assert(std::isfinite(v));
}

int main() {
    test_attention_shapes();
    std::printf("phase2 attention OK backend=%s\n", spider_backend_name());
    return 0;
}
