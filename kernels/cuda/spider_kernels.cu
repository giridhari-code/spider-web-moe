#include "spider_kernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) return (int)e; } while(0)

__global__ void router_kernel(const float* x,const float* bias,int B,int E,int K,int loop,
                              float* w,int32_t* idx) {
    int b=blockIdx.x; if(b>=B || threadIdx.x!=0) return;
    float mx=-CUDART_INF_F;
    for(int e=0;e<E;++e) mx=fmaxf(mx,x[b*E+e]+bias[loop*E+e]);
    float denom=0.f;
    for(int e=0;e<E;++e) denom+=expf(x[b*E+e]+bias[loop*E+e]-mx);
    float chosen=0.f;
    for(int k=0;k<K;++k) {
        int best=-1; float val=-1.f;
        for(int e=0;e<E;++e) {
            bool used=false; for(int q=0;q<k;++q) used |= idx[b*K+q]==e;
            float p=expf(x[b*E+e]+bias[loop*E+e]-mx)/denom;
            if(!used && p>val) {val=p;best=e;}
        }
        idx[b*K+k]=best; w[b*K+k]=val; chosen+=val;
    }
    for(int k=0;k<K;++k) w[b*K+k]/=fmaxf(chosen,1e-20f);
}

__global__ void graph_kernel(const float* h,int B,int H,int E,int K,
 const int32_t* rp,const int32_t* ci,const float* ew,int nnz,int loop,
 const int32_t* assign,const float* rw,float* out) {
    int b=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if(b>=B||j>=H) return;
    float acc=0.f;
    for(int k=0;k<K;++k) {
        int src=assign[b*K+k]; if(src<0||src>=E) continue;
        for(int e=rp[src];e<rp[src+1];++e) if(e<nnz)
            acc += ew[loop*nnz+e]*rw[b*K+k]*h[b*H+j];
    }
    out[b*H+j]=acc;
}

__global__ void expert_kernel(const float* x,int B,int H,int U,int E,int K,
 const int32_t* ids,const float* rw,const float* up,const float* down,float* out) {
    int b=blockIdx.x, j=blockIdx.y*blockDim.x+threadIdx.x;
    if(b>=B||j>=H) return;
    float total=0.f;
    for(int k=0;k<K;++k) {
        int e=ids[b*K+k]; if(e<0||e>=E) continue;
        float expert=0.f;
        for(int u=0;u<U;++u) {
            float a=0.f;
            for(int i=0;i<H;++i) a += x[b*H+i]*up[((e*H+i)*U)+u];
            float c=0.7978845608f*(a+0.044715f*a*a*a);
            float g=0.5f*a*(1.f+tanhf(c));
            expert += g*down[((e*U+u)*H)+j];
        }
        total += rw[b*K+k]*expert;
    }
    out[b*H+j]=total;
}

__global__ void epilogue_kernel(const float* h,const float* d,const float* x,int B,int H,
 float a,float beta,const float* gamma,const float* lb,float eps,float* out) {
    int b=blockIdx.x, tid=threadIdx.x;
    if(b>=B) return;
    extern __shared__ float buf[];
    // Thread zero computes stable Welford statistics. Each block owns one row.
    __shared__ float mean_s, inv_s;
    if(tid==0) {
        double mean=0.0,m2=0.0;
        for(int j=0;j<H;++j) {
            double u=(double)h[b*H+j]+a*d[b*H+j]+beta*x[b*H+j];
            double q=u-mean; mean+=q/(j+1); m2+=q*(u-mean);
        }
        mean_s=(float)mean; inv_s=rsqrtf((float)(m2/H)+eps);
    }
    __syncthreads();
    for(int j=tid;j<H;j+=blockDim.x) {
        float u=h[b*H+j]+a*d[b*H+j]+beta*x[b*H+j];
        out[b*H+j]=(u-mean_s)*inv_s*gamma[j]+lb[j];
    }
}

__global__ void exit_kernel(const float* h,const float* p,int B,int H,float threshold,uint8_t* mask) {
    int b=blockIdx.x; if(b>=B||threadIdx.x!=0) return;
    float dot=0.f; for(int j=0;j<H;++j) dot += h[b*H+j]*p[j];
    float conf=1.f/(1.f+expf(-dot)); mask[b]=(uint8_t)(conf>threshold);
}

extern "C" int spider_cuda_router_f32(const float*x,const float*b,int B,int E,int K,int loop,float*w,int32_t*i,void* stream) {
    if(!x||!b||!w||!i||B<=0||E<=0||K<=0||K>E||loop<0) return 1;
    router_kernel<<<B,128,0,(cudaStream_t)stream>>>(x,b,B,E,K,loop,w,i);
    return (int)cudaGetLastError();
}
extern "C" int spider_cuda_graph_message_passing_f32(const float*h,int B,int H,int E,int K,
 const int32_t*rp,const int32_t*ci,const float*ew,int nnz,int loop,const int32_t*ids,const float*rw,float*out,void*stream) {
    if(!h||!rp||!ci||!ew||!ids||!rw||!out||B<=0||H<=0||E<=0||K<=0||nnz<0||loop<0) return 1;
    graph_kernel<<<dim3(B,(H+127)/128),128,0,(cudaStream_t)stream>>>(h,B,H,E,K,rp,ci,ew,nnz,loop,ids,rw,out);
    return (int)cudaGetLastError();
}
extern "C" int spider_cuda_expert_grouped_gemm_f32(const float*x,int B,int H,int U,int E,int K,
 const int32_t*ids,const float*rw,const float*up,const float*down,float*out,void*stream) {
    if(!x||!ids||!rw||!up||!down||!out||B<=0||H<=0||U<=0||E<=0||K<=0) return 1;
    expert_kernel<<<dim3(B,(H+127)/128),128,0,(cudaStream_t)stream>>>(x,B,H,U,E,K,ids,rw,up,down,out);
    return (int)cudaGetLastError();
}
extern "C" int spider_cuda_loop_epilogue_f32(const float*h,const float*d,const float*x,int B,int H,float a,float beta,
 const float*g,const float*lb,float eps,float*out,void*stream) {
    if(!h||!d||!x||!g||!lb||!out||B<=0||H<=0||eps<=0) return 1;
    epilogue_kernel<<<B,256,((H+255)/256)*sizeof(float),(cudaStream_t)stream>>>(h,d,x,B,H,a,beta,g,lb,eps,out);
    return (int)cudaGetLastError();
}
extern "C" int spider_cuda_early_exit_f32(const float*h,const float*p,int B,int H,float threshold,uint8_t*mask,void*stream) {
    if(!h||!p||!mask||B<=0||H<=0) return 1;
    exit_kernel<<<B,128,0,(cudaStream_t)stream>>>(h,p,B,H,threshold,mask);
    return (int)cudaGetLastError();
}
