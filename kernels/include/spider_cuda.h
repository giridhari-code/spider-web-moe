#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int spider_cuda_router_f32(const float*,const float*,int,int,int,int,float*,int32_t*,void*);
int spider_cuda_graph_message_passing_f32(const float*,int,int,int,int,const int32_t*,const int32_t*,const float*,int,int,const int32_t*,const float*,float*,void*);
int spider_cuda_expert_grouped_gemm_f32(const float*,int,int,int,int,int,const int32_t*,const float*,const float*,const float*,float*,void*);
int spider_cuda_loop_epilogue_f32(const float*,const float*,const float*,int,int,float,float,const float*,const float*,float,float*,void*);
int spider_cuda_early_exit_f32(const float*,const float*,int,int,float,uint8_t*,void*);
#ifdef __cplusplus
}
#endif
