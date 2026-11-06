/* Phase 2.1.1: RoPE tables + apply on Q/K (interleaved pairs). */
#include "spider_kernels.h"
#include <cmath>
#include <cstring>

extern "C" int spider_rope_precompute_f32(
    int head_dim, int max_T, float theta, float* cos_out, float* sin_out) {
    if (!cos_out || !sin_out || head_dim <= 0 || max_T <= 0 || head_dim % 2 != 0 || theta <= 0.f)
        return 1;
    const int half = head_dim / 2;
    for (int t = 0; t < max_T; ++t) {
        for (int i = 0; i < half; ++i) {
            float freq = 1.f / std::pow(theta, (2.f * i) / static_cast<float>(head_dim));
            float ang = static_cast<float>(t) * freq;
            cos_out[t * half + i] = std::cos(ang);
            sin_out[t * half + i] = std::sin(ang);
        }
    }
    return 0;
}

extern "C" int spider_rope_apply_f32(
    float* x, int B, int n_heads, int T, int head_dim,
    const float* cos_table, const float* sin_table, int position) {
    if (!x || !cos_table || !sin_table || B <= 0 || n_heads <= 0 || T <= 0 || head_dim <= 0 ||
        head_dim % 2 != 0 || position < 0)
        return 1;
    const int half = head_dim / 2;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            for (int t = 0; t < T; ++t) {
                int abs_t = position + t;
                const float* c = cos_table + static_cast<size_t>(abs_t) * half;
                const float* s = sin_table + static_cast<size_t>(abs_t) * half;
                float* row = x + ((((size_t)b * n_heads + h) * T + t) * head_dim);
                for (int i = 0; i < half; ++i) {
                    float x0 = row[i];
                    float x1 = row[i + half];
                    row[i] = x0 * c[i] - x1 * s[i];
                    row[i + half] = x0 * s[i] + x1 * c[i];
                }
            }
        }
    }
    return 0;
}
