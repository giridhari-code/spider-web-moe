#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

static void fill_identity_qkv(float* Wq, float* Wo, int H) {
    std::memset(Wq, 0, sizeof(float) * 3 * H * H);
    std::memset(Wo, 0, sizeof(float) * H * H);
    for (int i = 0; i < H; ++i) {
        Wq[i * H + i] = 1.f;
        Wq[(H + i) * H + i] = 1.f;
        Wq[(2 * H + i) * H + i] = 1.f;
        Wo[i * H + i] = 1.f;
    }
}

static void test_prefill_then_decode() {
    const int B = 1, H = 8, nh = 2, max_T = 16;
    const int D = H / nh;
    std::vector<float> Wq(3 * H * H), Wo(H * H);
    fill_identity_qkv(Wq.data(), Wo.data(), H);

    size_t kv_sz = spider_kv_bytes(B, nh, max_T, D) / sizeof(float);
    std::vector<float> keys(kv_sz, 0.f), values(kv_sz, 0.f);
    SpiderKVCache cache;
    assert(spider_kv_init(&cache, keys.data(), values.data(), B, nh, max_T, D) == 0);

    const int Tp = 3;
    std::vector<float> xp(B * Tp * H, 0.2f);
    size_t ws = spider_attention_kv_workspace_bytes(B, Tp, H) / sizeof(float) + 8;
    std::vector<float> workspace(ws), outp(B * Tp * H);
    assert(spider_attention_causal_kv_f32(xp.data(), B, Tp, H, nh, Wq.data(), Wo.data(), &cache, 0,
                                         outp.data(), workspace.data()) == 0);
    assert(cache.length == 3);
    for (float v : outp) assert(std::isfinite(v));

    std::vector<float> xd(B * 1 * H, 0.3f), outd(B * 1 * H);
    ws = spider_attention_kv_workspace_bytes(B, 1, H) / sizeof(float) + 8;
    workspace.assign(ws);
    assert(spider_attention_causal_kv_f32(xd.data(), B, 1, H, nh, Wq.data(), Wo.data(), &cache, 3,
                                         outd.data(), workspace.data()) == 0);
    assert(cache.length == 4);
    for (float v : outd) assert(std::isfinite(v));

    assert(spider_kv_reset(&cache) == 0);
    assert(cache.length == 0);
}

static void test_capacity_error() {
    const int B = 1, H = 4, nh = 1, max_T = 2, D = 4;
    std::vector<float> keys(spider_kv_bytes(B, nh, max_T, D) / sizeof(float));
    std::vector<float> values(keys.size());
    SpiderKVCache cache;
    assert(spider_kv_init(&cache, keys.data(), values.data(), B, nh, max_T, D) == 0);
    std::vector<float> Wq(3 * H * H, 0.f), Wo(H * H, 0.f), x(B * 3 * H, 0.1f), out(B * 3 * H);
    std::vector<float> ws(spider_attention_kv_workspace_bytes(B, 3, H) / sizeof(float) + 4);
    int rc = spider_attention_causal_kv_f32(x.data(), B, 3, H, nh, Wq.data(), Wo.data(), &cache, 0,
                                            out.data(), ws.data());
    assert(rc == 3);
}

int main() {
    test_prefill_then_decode();
    test_capacity_error();
    std::printf("phase2.1 KV cache OK backend=%s\n", spider_backend_name());
    return 0;
}
