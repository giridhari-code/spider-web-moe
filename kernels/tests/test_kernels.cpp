#include "spider_kernels.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    // Router: normalized top-2 and valid expert IDs.
    const int B=2,E=4,K=2,H=8;
    float logits[B*E]={1,2,3,4, 4,3,2,1};
    float bias[E]={0,0,0,0}, weights[B*K]; int32_t ids[B*K];
    assert(spider_router_f32(logits,bias,B,E,K,0,weights,ids)==0);
    for(int b=0;b<B;++b) {
        assert(ids[b*K]>=0 && ids[b*K]<E);
        assert(ids[b*K+1]>=0 && ids[b*K+1]<E);
        assert(std::fabs(weights[b*K]+weights[b*K+1]-1.f)<1e-5f);
    }
    // CSR graph gather: one selected source, one edge, weighted hidden copy.
    const int32_t row_ptr[E+1]={0,1,1,1,1};
    const int32_t col_idx[1]={2};
    const float edge_w[1]={0.5f};
    const int32_t selected[B*K]={0,1, -1, -1};
    const float route[B*K]={0.25f,0.75f,0.f,0.f};
    float hidden[B*H], messages[B*H];
    for(int i=0;i<B*H;++i) hidden[i]=1.f;
    assert(spider_graph_message_passing_f32(hidden,B,H,E,K,row_ptr,col_idx,edge_w,1,0,
                                              selected,route,messages)==0);
    for(int j=0;j<H;++j) assert(std::fabs(messages[j]-0.125f)<1e-6f);

    // One-expert FFN path has finite output.
    const int32_t one_expert[B*K]={0,-1,0,-1};
    const float one_route[B*K]={1.f,0.f,1.f,0.f};
    std::vector<float> wup(E*H*4,0.01f), wdown(E*4*H,0.01f), expert_out(B*H);
    assert(spider_expert_grouped_gemm_f32(hidden,B,H,4,E,1,one_expert,one_route,
                                           wup.data(),wdown.data(),expert_out.data())==0);
    for(float v:expert_out) assert(std::isfinite(v));

    // Epilogue finite and normalized.
    float h[B*H],d[B*H],x[B*H],g[H],lb[H],out[B*H];
    for(int i=0;i<B*H;++i) h[i]=0.01f*i,d[i]=0.02f*i,x[i]=0.03f*i;
    for(int i=0;i<H;++i) g[i]=1.f,lb[i]=0.f;
    assert(spider_loop_epilogue_f32(h,d,x,B,H,.5f,.1f,g,lb,1e-5f,out)==0);
    for(float v:out) assert(std::isfinite(v));
    float proj[H]; for(float &v:proj)v=.1f; uint8_t mask[B];
    assert(spider_early_exit_f32(out,proj,B,H,.85f,mask)==0);
    std::printf("backend=%s; router, epilogue, exit: OK\\n",spider_backend_name());
}
