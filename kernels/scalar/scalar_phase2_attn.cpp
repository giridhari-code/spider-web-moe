/* Phase 2.0: causal multi-head self-attention (scalar, no RoPE, no KV cache). */
#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" size_t spider_attention_workspace_bytes(int B, int T, int H, int n_heads) {
    if (B <= 0 || T <= 0 || H <= 0 || n_heads <= 0) return 0;
    return (size_t)B * (size_t)T * (size_t)(3 * H) * sizeof(float);
}

extern "C" int spider_attention_causal_f32(
    const float* x, int B, int T, int H, int n_heads,
    const float* W_qkv, const float* W_o,
    float* out, float* workspace) {
    if (!x || !W_qkv || !W_o || !out || !workspace || B <= 0 || T <= 0 || H <= 0 ||
        n_heads <= 0 || H % n_heads != 0)
        return 1;
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

    std::vector<float> ctx(static_cast<size_t>(B) * T * H, 0.f);
    std::vector<float> scores(static_cast<size_t>(T) * T);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            for (int qi = 0; qi < T; ++qi) {
                float mx = -std::numeric_limits<float>::infinity();
                for (int kj = 0; kj <= qi; ++kj) {
                    double dot = 0.0;
                    const float* q = qkv + (static_cast<size_t>(b) * T + qi) * (3 * H) + h * D;
                    const float* k =
                        qkv + (static_cast<size_t>(b) * T + kj) * (3 * H) + H + h * D;
                    for (int d = 0; d < D; ++d) dot += static_cast<double>(q[d]) * k[d];
                    float s = static_cast<float>(dot) * scale;
                    scores[qi * T + kj] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0.f;
                for (int kj = 0; kj <= qi; ++kj) {
                    float e = std::exp(scores[qi * T + kj] - mx);
                    scores[qi * T + kj] = e;
                    sum += e;
                }
                for (int kj = 0; kj <= qi; ++kj)
                    scores[qi * T + kj] /= std::max(sum, 1e-20f);

                float* cout = ctx.data() + (static_cast<size_t>(b) * T + qi) * H + h * D;
                for (int d = 0; d < D; ++d) cout[d] = 0.f;
                for (int kj = 0; kj <= qi; ++kj) {
                    const float* v =
                        qkv + (static_cast<size_t>(b) * T + kj) * (3 * H) + 2 * H + h * D;
                    float a = scores[qi * T + kj];
                    for (int d = 0; d < D; ++d) cout[d] += a * v[d];
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
