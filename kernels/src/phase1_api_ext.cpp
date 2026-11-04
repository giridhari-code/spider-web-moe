/* Phase-1 API extensions: f16 stubs + caps. moe_loop_step lives in scalar_phase1_1.cpp */
#include "spider_kernels.h"

extern "C" int spider_router_f16(const uint16_t*, const uint16_t*, int, int, int, int, float*, int32_t*) {
    return -2;
}
extern "C" int spider_graph_message_passing_f16(const uint16_t*,int,int,int,int,const int32_t*,const int32_t*,const float*,int,int,const int32_t*,const float*,float*) {
    return -2;
}
extern "C" int spider_expert_grouped_gemm_f16(const float*,int,int,int,int,int,const int32_t*,const float*,const uint16_t*,const uint16_t*,float*) {
    return -2;
}
extern "C" int spider_loop_epilogue_f16(const float*,const float*,const float*,int,int,float,float,const float*,const float*,float,float*) {
    return -2;
}

extern "C" uint32_t spider_backend_caps(void) {
    return 1u; /* f32; bit2/4 when CUDA/NEON paths expose more */
}
