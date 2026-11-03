#include "spider_kernels.h"
extern "C" int spider_scalar_router_f32(const float*,const float*,int,int,int,int,float*,int32_t*);
extern "C" int spider_scalar_graph_message_passing_f32(const float*,int,int,int,int,const int32_t*,const int32_t*,const float*,int,int,const int32_t*,const float*,float*);
extern "C" int spider_scalar_expert_grouped_gemm_f32(const float*,int,int,int,int,int,const int32_t*,const float*,const float*,const float*,float*);
extern "C" int spider_scalar_loop_epilogue_f32(const float*,const float*,const float*,int,int,float,float,const float*,const float*,float,float*);
extern "C" int spider_scalar_early_exit_f32(const float*,const float*,int,int,float,uint8_t*);
extern "C" int spider_router_f32(const float*a,const float*b,int c,int d,int e,int f,float*g,int32_t*h){return spider_scalar_router_f32(a,b,c,d,e,f,g,h);}
extern "C" int spider_graph_message_passing_f32(const float*a,int b,int c,int d,int e,const int32_t*f,const int32_t*g,const float*h,int i,int j,const int32_t*k,const float*l,float*m){return spider_scalar_graph_message_passing_f32(a,b,c,d,e,f,g,h,i,j,k,l,m);}
extern "C" int spider_expert_grouped_gemm_f32(const float*a,int b,int c,int d,int e,int f,const int32_t*g,const float*h,const float*i,const float*j,float*k){return spider_scalar_expert_grouped_gemm_f32(a,b,c,d,e,f,g,h,i,j,k);}
extern "C" int spider_loop_epilogue_f32(const float*a,const float*b,const float*c,int d,int e,float f,float g,const float*h,const float*i,float j,float*k){return spider_scalar_loop_epilogue_f32(a,b,c,d,e,f,g,h,i,j,k);}
extern "C" int spider_early_exit_f32(const float*a,const float*b,int c,int d,float e,uint8_t*f){return spider_scalar_early_exit_f32(a,b,c,d,e,f);}
extern "C" const char* spider_backend_name(void){return "scalar";}
