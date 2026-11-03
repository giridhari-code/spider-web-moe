#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

extern "C" int spider_scalar_router_f32(const float* logits, const float* bias, int B, int E, int K,
                                  int loop_idx, float* weights, int32_t* indices) {
    if (!logits || !bias || !weights || !indices || B <= 0 || E <= 0 || K <= 0 || K > E || loop_idx < 0) return 1;
    const float* b = bias + static_cast<size_t>(loop_idx) * E;
    for (int n=0; n<B; ++n) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int e=0;e<E;++e) mx = std::max(mx, logits[n*E+e] + b[e]);
        float sum=0.f;
        for (int e=0;e<E;++e) sum += std::exp(logits[n*E+e] + b[e] - mx);
        // Deterministic partial top-k; selected probabilities are renormalized.
        float selected_sum=0.f;
        for (int k=0;k<K;++k) {
            int best=-1; float bestv=-1.f;
            for (int e=0;e<E;++e) {
                bool used=false;
                for (int q=0;q<k;++q) used |= indices[n*K+q] == e;
                if (used) continue;
                float p=std::exp(logits[n*E+e] + b[e] - mx)/sum;
                if (p>bestv) { bestv=p; best=e; }
            }
            indices[n*K+k]=best;
            weights[n*K+k]=bestv;
            selected_sum += bestv;
        }
        for (int k=0;k<K;++k) weights[n*K+k] /= std::max(selected_sum, 1e-20f);
    }
    return 0;
}

extern "C" int spider_scalar_graph_message_passing_f32(
    const float* hidden, int B, int H, int E, int K, const int32_t* row_ptr,
    const int32_t* col_idx, const float* edge_weights, int nnz, int loop_idx,
    const int32_t* assign, const float* route_weights, float* messages) {
    if (!hidden || !row_ptr || !col_idx || !edge_weights || !assign || !route_weights || !messages ||
        B<=0 || H<=0 || E<=0 || K<=0 || nnz<0 || loop_idx<0 || row_ptr[0]!=0 || row_ptr[E]!=nnz) return 1;
    std::fill(messages, messages + static_cast<size_t>(B)*H, 0.f);
    const float* ew=edge_weights + static_cast<size_t>(loop_idx)*nnz;
    for (int b=0;b<B;++b) for (int k=0;k<K;++k) {
        int src=assign[b*K+k]; if (src<0) continue; if (src>=E) return 2;
        for (int edge=row_ptr[src];edge<row_ptr[src+1];++edge) {
            int dst=col_idx[edge]; if (dst<0 || dst>=E) return 2;
            float scale=ew[edge]*route_weights[b*K+k];
            for (int j=0;j<H;++j) messages[b*H+j] += scale*hidden[b*H+j];
        }
    }
    return 0;
}

extern "C" int spider_scalar_expert_grouped_gemm_f32(
    const float* tokens, int B, int H, int U, int E, int K,
    const int32_t* assign, const float* route, const float* Wup, const float* Wdown, float* output) {
    if (!tokens || !assign || !route || !Wup || !Wdown || !output || B<=0 || H<=0 || U<=0 || E<=0 || K<=0) return 1;
    std::fill(output, output+static_cast<size_t>(B)*H, 0.f);
    // W_up [E,H,U], W_down [E,U,H], expert-major contiguous.
    for (int b=0;b<B;++b) for (int k=0;k<K;++k) {
        int e=assign[b*K+k]; if (e<0) continue; if (e>=E) return 2;
        for (int u=0;u<U;++u) {
            float a=0.f;
            for (int i=0;i<H;++i) a += tokens[b*H+i]*Wup[(static_cast<size_t>(e)*H+i)*U+u];
            // GELU (tanh approximation).
            float x=a, c=0.7978845608f*(x+0.044715f*x*x*x);
            float activated=0.5f*x*(1.f+std::tanh(c));
            for (int j=0;j<H;++j) output[b*H+j] += route[b*K+k]*activated*Wdown[(static_cast<size_t>(e)*U+u)*H+j];
        }
    }
    return 0;
}

extern "C" int spider_scalar_loop_epilogue_f32(
    const float* h, const float* d, const float* x0, int B, int H, float alpha, float beta,
    const float* gamma, const float* ln_bias, float eps, float* out) {
    if (!h || !d || !x0 || !gamma || !ln_bias || !out || B<=0 || H<=0 || eps<=0) return 1;
    for (int b=0;b<B;++b) {
        double mean=0.0, m2=0.0;
        for (int j=0;j<H;++j) {
            double u=h[b*H+j]+alpha*d[b*H+j]+beta*x0[b*H+j];
            double delta=u-mean; mean += delta/(j+1); m2 += delta*(u-mean);
        }
        float inv=1.f/std::sqrt(static_cast<float>(m2/H)+eps);
        for (int j=0;j<H;++j) {
            float u=h[b*H+j]+alpha*d[b*H+j]+beta*x0[b*H+j];
            out[b*H+j]=(u-static_cast<float>(mean))*inv*gamma[j]+ln_bias[j];
        }
    }
    return 0;
}

extern "C" int spider_scalar_early_exit_f32(const float* h, const float* proj, int B, int H,
                                     float threshold, uint8_t* exit_mask) {
    if (!h || !proj || !exit_mask || B<=0 || H<=0) return 1;
    for (int b=0;b<B;++b) {
        double dot=0.; for (int j=0;j<H;++j) dot += static_cast<double>(h[b*H+j])*proj[j];
        float z=static_cast<float>(dot);
        float conf = z >= 0.f ? 1.f/(1.f+std::exp(-z)) : std::exp(z)/(1.f+std::exp(z));
        exit_mask[b]=static_cast<uint8_t>(conf>threshold);
    }
    return 0;
}

extern "C" const char* spider_scalar_backend_name(void) { return "scalar"; }
