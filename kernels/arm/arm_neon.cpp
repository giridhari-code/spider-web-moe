#include "spider_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define SPIDER_HAS_NEON 1
#else
  #define SPIDER_HAS_NEON 0
#endif

extern "C" int spider_scalar_router_f32(const float*, const float*, int, int, int, int, float*, int32_t*);
extern "C" int spider_scalar_graph_message_passing_f32(const float*,int,int,int,int,const int32_t*,const int32_t*,const float*,int,int,const int32_t*,const float*,float*);
extern "C" int spider_scalar_expert_grouped_gemm_f32(const float*,int,int,int,int,int,const int32_t*,const float*,const float*,const float*,float*);
extern "C" int spider_scalar_loop_epilogue_f32(const float*,const float*,const float*,int,int,float,float,const float*,const float*,float,float*);
extern "C" int spider_scalar_early_exit_f32(const float*,const float*,int,int,float,uint8_t*);

#if SPIDER_HAS_NEON
extern "C" int spider_loop_epilogue_neon_f32(
 const float* h,const float* d,const float* x0,int B,int H,float alpha,float beta,
 const float* gamma,const float* bias,float eps,float* out) {
    if(!h||!d||!x0||!gamma||!bias||!out||B<=0||H<=0||eps<=0) return 1;
    const float32x4_t av=vdupq_n_f32(alpha), bv=vdupq_n_f32(beta);
    for(int b=0;b<B;++b) {
        // Welford update in FP64 scalar for stable statistics; NEON accelerates affine/residual pass.
        double mean=0.,m2=0.;
        for(int j=0;j<H;++j) { double u=h[b*H+j]+alpha*d[b*H+j]+beta*x0[b*H+j];
            double q=u-mean; mean+=q/(j+1); m2+=q*(u-mean); }
        float inv=1.f/std::sqrt(static_cast<float>(m2/H)+eps);
        int j=0; const float32x4_t mv=vdupq_n_f32(static_cast<float>(mean)), iv=vdupq_n_f32(inv);
        for(;j+4<=H;j+=4) {
            auto hv=vld1q_f32(h+b*H+j), dv=vld1q_f32(d+b*H+j), xv=vld1q_f32(x0+b*H+j);
            auto u=vmlaq_f32(hv,av,dv); u=vmlaq_f32(u,bv,xv);
            auto g=vld1q_f32(gamma+j), z=vld1q_f32(bias+j);
            auto y=vmulq_f32(vmulq_f32(vsubq_f32(u,mv),iv),g);
            vst1q_f32(out+b*H+j,vaddq_f32(y,z));
        }
        for(;j<H;++j) { float u=h[b*H+j]+alpha*d[b*H+j]+beta*x0[b*H+j];
            out[b*H+j]=(u-static_cast<float>(mean))*inv*gamma[j]+bias[j]; }
    }
    return 0;
}
extern "C" int spider_early_exit_neon_f32(const float* h,const float* p,int B,int H,float threshold,uint8_t* mask) {
    if(!h||!p||!mask||B<=0||H<=0) return 1;
    for(int b=0;b<B;++b) {
        float32x4_t acc=vdupq_n_f32(0.f); int j=0;
        for(;j+4<=H;j+=4) acc=vmlaq_f32(acc,vld1q_f32(h+b*H+j),vld1q_f32(p+j));
        float dot=vaddvq_f32(acc); for(;j<H;++j) dot+=h[b*H+j]*p[j];
        float conf=dot>=0?1.f/(1.f+std::exp(-dot)):std::exp(dot)/(1.f+std::exp(dot));
        mask[b]=static_cast<uint8_t>(conf>threshold);
    }
    return 0;
}
#endif

// Until dedicated vector kernels are added, the remaining operations use the scalar
// reference implementation. This keeps ARM correctness while SIMD work is incremental.
extern "C" int spider_router_f32(const float*a,const float*b,int c,int d,int e,int f,float*g,int32_t*h){return spider_scalar_router_f32(a,b,c,d,e,f,g,h);}
extern "C" int spider_graph_message_passing_f32(const float*a,int b,int c,int d,int e,const int32_t*f,const int32_t*g,const float*h,int i,int j,const int32_t*k,const float*l,float*m){return spider_scalar_graph_message_passing_f32(a,b,c,d,e,f,g,h,i,j,k,l,m);}
extern "C" int spider_expert_grouped_gemm_f32(const float*a,int b,int c,int d,int e,int f,const int32_t*g,const float*h,const float*i,const float*j,float*k){return spider_scalar_expert_grouped_gemm_f32(a,b,c,d,e,f,g,h,i,j,k);}
extern "C" int spider_loop_epilogue_f32(const float*a,const float*b,const float*c,int d,int e,float f,float g,const float*h,const float*i,float j,float*k){
#if SPIDER_HAS_NEON
 return spider_loop_epilogue_neon_f32(a,b,c,d,e,f,g,h,i,j,k);
#else
 return spider_scalar_loop_epilogue_f32(a,b,c,d,e,f,g,h,i,j,k);
#endif
}
extern "C" int spider_early_exit_f32(const float*a,const float*b,int c,int d,float e,uint8_t*f){
#if SPIDER_HAS_NEON
 return spider_early_exit_neon_f32(a,b,c,d,e,f);
#else
 return spider_scalar_early_exit_f32(a,b,c,d,e,f);
#endif
}
extern "C" const char* spider_backend_name(void) {
#if SPIDER_HAS_NEON
 return "arm-neon";
#else
 return "arm-scalar-fallback";
#endif
}
