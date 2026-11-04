/* Phase-1 API extensions — merge into scalar/dispatch or compile alongside. */
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

extern "C" int spider_moe_loop_step_f32(
    const float* h, const float* x0, int B, int H, int E, int K, int U, int loop_idx,
    const float* router_w, const float* loop_bias,
    const int32_t* row_ptr, const int32_t* col_idx, const float* edge_w, int nnz,
    const float* W_up, const float* W_down,
    const float* ln_weight, const float* ln_bias, float alpha, float beta, float eps,
    const float* exit_proj, float exit_threshold,
    float* h_next, uint8_t* exit_mask,
    int32_t* tmp_assign, float* tmp_route, float* tmp_messages) {
    (void)h; (void)x0; (void)B; (void)H; (void)E; (void)K; (void)U; (void)loop_idx;
    (void)router_w; (void)loop_bias; (void)row_ptr; (void)col_idx; (void)edge_w; (void)nnz;
    (void)W_up; (void)W_down; (void)ln_weight; (void)ln_bias; (void)alpha; (void)beta; (void)eps;
    (void)exit_proj; (void)exit_threshold; (void)h_next; (void)exit_mask;
    (void)tmp_assign; (void)tmp_route; (void)tmp_messages;
    return -3; /* Python NativeBackend composes in Phase 1 */
}

extern "C" uint32_t spider_backend_caps(void) {
    return 1u;
}
