/* Phase 2.1: incremental KV-cache causal attention (scalar). */
#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" size_t spider_kv_bytes(int B, int n_heads, int max_T, int head_dim) {
    if (B <= 0 || n_heads <= 0 || max_T <= 0 || head_dim <= 0) return 0;
    return (size_t)B * (size_t)n_heads * (size_t)max_T * (size_t)head_dim * sizeof(float);
}

extern "C" int spider_kv_init(SpiderKVCache* cache, float* keys, float* values,
                              int B, int n_heads, int max_T, int head_dim) {
    if (!cache || !keys || !values || B <= 0 || n_heads <= 0 || max_T <= 0 || head_dim <= 0)
        return 1;
    cache->keys = keys;
    cache->values = values;
    cache->B = B;
    cache->n_heads = n_heads;
    cache->max_T = max_T;
    cache->head_dim = head_dim;
    cache->length = 0;
    return 0;
}

extern "C" int spider_kv_reset(SpiderKVCache* cache) {
    if (!cache) return 1;
    cache->length = 0;
    return 0;
}

extern "C" size_t spider_attention_kv_workspace_bytes(int B, int T, int H) {
    if (B <= 0 || T <= 0 || H <= 0) return 0;
    return (size_t)B * (size_t)T * (size_t)(3 * H) * sizeof(float);
}

static inline float* kv_ptr(float* base, int b, int h, int t, int d,
                            int n_heads, int max_T, int head_dim) {
    return base + ((((size_t)b * n_heads + h) * max_T + t) * head_dim + d);
}

extern "C" int spider_attention_causal_kv_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    SpiderKVCache* cache, int position,
    float* out, float* workspace) {
    if (!x || !W_qkv || !W_o || !cache || !out || !workspace || B <= 0 || T <= 0 || H <= 0 ||
        n_heads <= 0 || H % n_heads != 0 || position < 0)
        return 1;
    if (cache->B != B || cache->n_heads != n_heads || cache->head_dim != H / n_heads)
        return 2;
    if (position + T > cache->max_T) return 3;

    const int D = H / n_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    float* qkv = workspace;

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* xin = x + (static_cast<size_t>(b) * T + t) * H;
            float* row = qkv + (static_cast<size_t>(b) * T + t) * (3 * H);
            for (int o = 0; o < 3 * H; ++o) {
                const float* wrow = W_qkv + static_cast<size_t>(o) * H;
                double acc = 0.0;
                for (int i = 0; i < H; ++i) acc += static_cast<double>(xin[i]) * wrow[i];
                row[o] = static_cast<float>(acc);
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int abs_t = position + t;
            const float* row = qkv + (static_cast<size_t>(b) * T + t) * (3 * H);
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < D; ++d) {
                    *kv_ptr(cache->keys, b, h, abs_t, d, n_heads, cache->max_T, D) =
                        row[H + h * D + d];
                    *kv_ptr(cache->values, b, h, abs_t, d, n_heads, cache->max_T, D) =
                        row[2 * H + h * D + d];
                }
            }
        }
    }
    int new_len = position + T;
    if (new_len > cache->length) cache->length = new_len;

    std::vector<float> ctx(static_cast<size_t>(B) * T * H, 0.f);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            for (int ti = 0; ti < T; ++ti) {
                int abs_q = position + ti;
                const float* q =
                    qkv + (static_cast<size_t>(b) * T + ti) * (3 * H) + h * D;

                float mx = -std::numeric_limits<float>::infinity();
                std::vector<float> sc(static_cast<size_t>(abs_q + 1));
                for (int kj = 0; kj <= abs_q; ++kj) {
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) {
                        float kd = *kv_ptr(cache->keys, b, h, kj, d, n_heads, cache->max_T, D);
                        dot += static_cast<double>(q[d]) * kd;
                    }
                    float s = static_cast<float>(dot) * scale;
                    sc[kj] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0.f;
                for (int kj = 0; kj <= abs_q; ++kj) {
                    sc[kj] = std::exp(sc[kj] - mx);
                    sum += sc[kj];
                }
                for (int kj = 0; kj <= abs_q; ++kj) sc[kj] /= std::max(sum, 1e-20f);

                float* cout = ctx.data() + (static_cast<size_t>(b) * T + ti) * H + h * D;
                for (int d = 0; d < D; ++d) cout[d] = 0.f;
                for (int kj = 0; kj <= abs_q; ++kj) {
                    float a = sc[kj];
                    for (int d = 0; d < D; ++d) {
                        float vd = *kv_ptr(cache->values, b, h, kj, d, n_heads, cache->max_T, D);
                        cout[d] += a * vd;
                    }
                }
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* cin = ctx.data() + (static_cast<size_t>(b) * T + t) * H;
            float* orow = out + (static_cast<size_t>(b) * T + t) * H;
            for (int j = 0; j < H; ++j) {
                const float* wrow = W_o + static_cast<size_t>(j) * H;
                double acc = 0.0;
                for (int i = 0; i < H; ++i) acc += static_cast<double>(cin[i]) * wrow[i];
                orow[j] = static_cast<float>(acc);
            }
        }
    }
    return 0;
}
