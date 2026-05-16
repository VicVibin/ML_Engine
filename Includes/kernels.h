#pragma once
#include <curand_kernel.h>
#include <string>
#include <cstdint>
#include <cfloat>

static constexpr int BLOCK_SIZE = 32;
static constexpr int THREADSPERBLOCK = 256;
static constexpr float PIBY2  = 1.57079632679;
static constexpr float PI  = 3.14159265359;
using str = std::string;

template <typename T>
__global__ void fillKernel(T* data, T value, long long n) 
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx  >= n) return;
    data[idx] = value;
}

template <typename T>
void Zerograd(const str name, T* ptr, const long long total)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total+tpb-1)/tpb;
    fillKernel<<<bpg,tpb>>>(ptr, (T)0.0f, total);
    //CheckError("Zero_grad " + name);
}

__device__ double atomicAddDouble(double* __restrict__ address, double val);

__device__ inline void atomicMaxFloat(float* __restrict__  addr, float value);

__device__ inline void atomicMinFloat(float* __restrict__ addr, float value);

__device__ __forceinline__ double warpReduceSum(double val, int offset = 16);

__device__ __forceinline__ int batch_chan_offset(int flag, int batch_idx, int channel_idx, int num_channels, int inner_size);

__device__ int ceil_div(const int a, const int b);

__global__ void permute(const float* __restrict__ X, float* __restrict__ Y, int d0, int d1, int d2, int d3, int i0, int i1, int i2, int i3);

__global__ void mulKernel(const float*  X, const float* Y, float*  Z, const long long total_size, const bool deriv = false);

__global__ void HeadifyColChannel(const float* __restrict__ input, float* __restrict__ output, const int batch, const int req_channels, const int row, const int col, int flag);

__global__ void PEncoding(float* __restrict__ X, const int t, const int t_dim, const int total_size);

__global__ void MatPEncoding(const float* __restrict__ X, float* __restrict__ output, const int batch, const int row, const int col, const int total, const int start_idx = 0);


__global__ void bmm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int batch_size, int m, int n, int p, int backward = 0, int A=1, int B=1, int C=1, float scale = 1);

__global__ void bmmABT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int batch_size, int m, int n, int p, int backward = 0,int A=1, int B=1, int C=1, float scale = 1);

__global__ void bmmATB(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int batch_size, int m, int n, int p, int backward = 0,int A=1, int B=1, int C=1, float scale = 1);

__global__ void bmmATBT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int batch_size, int m, int n, int p,int backward = 0,int A=1,int B=1, int C=1, float scale = 1);


__global__ void bcmm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, const  int batch_size, const int num_channels, const int m, const int n, 
                    const int p, const int backward = 0, const int A = 3, const int B = 3, const int C = 3, const float scale = 1);

__global__ void bcmmABT(const float* __restrict__ a, const float* __restrict__ b,float* __restrict__ c, const  int batch_size, const int num_channels, const int m, const int n, const int p,
                    const int backward = 0, const int A = 3, const int B = 3, const int C = 3, const float scale = 1);

__global__ void bcmmATB(const float* __restrict__ a, const float* __restrict__ b,float* __restrict__ c, const  int batch_size, const int num_channels, const int m, const int n, const int p,
                    const int backward = 0, const int A = 3, const int B = 3, const int C = 3, const float scale = 1);

__global__ void bcmmATBT(const float* __restrict__ a, const float* __restrict__ b,float* __restrict__ c, const  int batch_size, const int num_channels, const int m, const int n, const int p,
                    const int backward = 0, const int A = 3, const int B = 3, const int C = 3, const float scale = 1);


__global__ void BatchMean(const float* __restrict__ data, float* __restrict__ mean, const int batch, const int channels, const int row, const int col, const bool scale = true);

__global__ void LayerMean(const float* __restrict__ data, float* __restrict__ mean, const int batch, const int channels, const int row, const int col, const bool scale = true);

__global__ void GroupMean(const float* __restrict__ data, float* __restrict__ mean,const int batch,const int channels,const int groups, const int row, const int col, const bool scale = true);

__global__ void InstanceMean(const float* __restrict__ data, float* __restrict__ mean,const int batch,const int channels,const int row,const int col, const bool scale = true);


__global__ void BatchStd(const float* __restrict__ data,const float* __restrict__ mean,float* __restrict__ std,const int batch, const int channels, const int row, const int col);

__global__ void LayerStd(const float* __restrict__ data,const float* __restrict__ mean,float* __restrict__ std,const int batch, const int channels, const int row, const int col);

__global__ void GroupStd(const float* __restrict__ data,const float* __restrict__ mean,float* __restrict__ std,const int batch,const int channels,const int groups, const int row, const int col);

__global__ void InstanceStd(const float* __restrict__ data,const float* __restrict__ mean,float* __restrict__ std,const int batch,const int channels, const int row, const int col);



__global__ void LNorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ std,const int batch,const int channels,
                      const int row,const int col,const float gamma,const float beta,const float epsilon);

__global__ void BNorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ std,const int batch,const int channels,
                      const int row, const int col,const float gamma,const float beta, const float epsilon);

__global__ void GNorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ std,const int batch,const int channels, 
                      const int groups, const int row, const int col, const float gamma, const float beta, const float epsilon); 

__global__ void INorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ std,const int batch,const int channels, 
                      const int row,const int col,const float gamma, const float beta, const float epsilon); 


                      
__global__ void LayerBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma,const float* __restrict__ ggammanode, const float* __restrict__ variance, 
                              const float gamma, const float epsilon, const int batch, const int channels, const int row, const int col);

__global__ void BatchBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma,const float* __restrict__ ggammanode, 
                              const float* __restrict__ variance, const float gamma, const float epsilon,
                              const int batch, const int channels, const int row, const int col);

__global__ void GroupBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma, const float* __restrict__ ggammanode, 
                              const float* __restrict__ variance, const float gamma, const float epsilon, const int batch, 
                              const int channels, const int groups, const int row, const int col);

__global__ void InstanceBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma,const float* __restrict__ ggammanode, 
                              const float* __restrict__ variance, const float gamma, const float epsilon, const int batch, 
                              const int channels, const int row, const int col);

__global__ void LayerMeanGrad(const float* __restrict__ ngrad, float* __restrict__ igrad, const int batch, const int channels, const int row, const int col);

__global__ void BatchMeanGrad(const float* __restrict__ ngrad, float* __restrict__ igrad, const int batch, const int channels, const int row, const int col);

__global__ void dropoutKernel(const float* __restrict__ data, float* __restrict__ mask, float* __restrict__ output, const long long size, const float p, const uint64_t seed, const int deriv =0);

__global__ void Standard_Weights(float* __restrict__ w, const long long size, const float scale, const uint64_t seed);

__global__ void PadOutput(const float* __restrict__ input, float* __restrict__ output, const int batch, const int channels, 
                          const int inH, const int inW, const int outH, const int outW, const int padTop, const int padLeft);

__global__ void CV2D(const float* __restrict__ X, const float* __restrict__ K, const float* __restrict__ bias, float* __restrict__ Y, 
                     const int N, const int Cout, const int Cin, const int H, const int W, const int KH=3, const int KW=3, const int pad=0, const int stride=1);

__global__ void GV2D(const float* __restrict__ X, const float* __restrict__ dZ, float* __restrict__ dK, const int batch, 
                     const int out,const int in,const int a,const int b,const int c,const int d,const int pad,const int stride);

__global__ void CV2D_GradInput(const float* __restrict__ dY,const float* __restrict__ K, float* __restrict__ dX,
                               const int batch, const int out, const int inp, const int a, const int b,const int c,
                               const int d, const int outR, const int outC, const int pad,  const int stride);

__global__ void CVT2D(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, 
                      float* __restrict__ output, const int batch, const int out_channels,  const int inp_channels,  
                      const int inp_h, const int inp_w, const int kernel_h, const int kernel_w, const int pad, const int stride);

__global__ void GVT2D(const float* __restrict__ input,const float* __restrict__ grad_output,float* __restrict__ grad_weights,
                      const int batch, const int out_channels, const int inp_channels,const int inp_h, const int inp_w,
                      const int kernel_h, const int kernel_w,const int pad, const int stride);

__global__ void CVT2D_GradInput(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input, 
                                const int batch, const  int out_channels, const int inp_channels, const int inp_h, const int inp_w, 
                                const int kernel_h, const int kernel_w, const int pad, const int stride);

__global__ void ReLU(const float* __restrict__ input, float* __restrict__ output, const int total_size);

__global__ void deriv_ReLU(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size);

__global__ void LeakyReLU(const float* __restrict__ input, float* __restrict__ output, const int total_size);

__global__ void deriv_LeakyReLU(const float* __restrict__ X, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size);

__global__ void SiLU(const float* __restrict__ input, float* __restrict__ output, const int total_size);

__global__ void deriv_SiLU(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size);

__global__ void TaNH(const float* __restrict__ input, float* __restrict__ output, const int total_size);

__global__ void deriv_TaNH(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size);

__global__ void Sigmoid(const float* __restrict__ input, float* __restrict__ output, const int total_size);

__global__ void deriv_Sigmoid(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size);

__global__ void Channel_Concat(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int batch, const int c1, const int c2, const int row, const int col);

__global__ void Channel_Split(float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C, const int batch, const int c1, const int c2, const int row, const int col);

__global__ void Column_Concat(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int batch, const int channel, const int row, const int c1, const int c2);

__global__ void Column_Split(float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C, const int batch, const int channel, const int row, const int c1, const int c2);

__global__ void Vector_Concat(const float* __restrict__ A, float* __restrict__ B, const int batch, const int channel,
                             const int row, const int a_col, const int total_col, const int col_offset);

__global__ void Vector_Split(const float* __restrict__ dB, float* __restrict__ dA, const int batch, const int channel,
                       const int row, const int a_col, const int total_col, const int col_offset);

__global__ void transpose(const float* __restrict__ X, float* __restrict__ output, const int batch_size, const int channels, 
                          const int row, const int col, const int grad = 0);

__global__ void CopynCrop(const float* __restrict__ X, float* __restrict__ Y, const int batch_size, const int depth, const int a, const int b, const int c, const int d);

__global__ void PaddingCrop(const float* __restrict__ X, float* __restrict__ Y, const int batch_size,const int depth,const int a, const int b, const int c, const int d);

__global__ void SumSquared(double* __restrict__ scale, const float* __restrict__ grad, const long long total_size);

template <typename T>
__global__ void Compare(const T* __restrict__ scale, const T* __restrict__ scale_test, int total)
{   
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx >= total) return;
    const T a = scale[idx];
    const T b = scale_test[idx];
    if(a != b) 
    {
        //printf("Output is not the same at idx: %i, values are scale[idx] : %f and scale_test[idx]: %f", idx, a, b);
        __trap();
        return;
    }
}

__global__ void AdamUpdate(float* __restrict__ output, const float* __restrict__ grad, const int total_size, const int t, float* __restrict__ m, float* __restrict__ v,
                           const float b1, const float b2, const float epsilon, const float lr);

__global__ void AdamWUpdate(float* __restrict__ output, const float* __restrict__ grad, const int total_size, const int t, float* __restrict__ m, float* __restrict__ v,
                           const float b1, const float b2, const float epsilon, const float lambda, const float lr);

__global__ void KeyUpdate(float* __restrict__ EmbedSpace, const float* __restrict__ grad, const int* __restrict__ keys,const int clen, const int context_len, 
                          const int embed_dim, const float lr, const long long total);

__global__ void OneHotEmbeddings(float* __restrict__ output, const int* __restrict__ keys,int clen,int max_clen,int vocab_size,long long total_size);

__global__ void BCompress(const float* __restrict__ X, float* __restrict__ Y, int batch,int channels, int rows, int columns, const float scale = 1.0f);

__global__ void BCumAdd(float* __restrict__ X, const float* __restrict__ Y, int batch, int channels, int rows, int columns);

__global__ void broadcast_add_general(const float* __restrict__ X, const float* __restrict__ bias, float* __restrict__ output, const int batch_size, const int channels, 
                                      const int a, const int b, const int c, const int d);

__global__ void broadcast_add_backward(const float* __restrict__ grad_out, float* __restrict__ grad_B, const int batch_size, const int channels, const int a, const int b, const int c, const int d);

__global__ void broadcast_add(const float*  __restrict__ X, const float*  __restrict__ bias, float*  __restrict__ output, const int batch_size, const int kernels, const int row, const int col);

__global__ void Accumulate(const float* __restrict__ X, float* __restrict__ Y, const long long total_size, const double scale = 1.0);

__global__ void ScaleAdd(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ Z, const float scale, const long long total_size);

__global__ void Update(float* __restrict__ output, const float*  __restrict__ grad, const int total_size, const double* __restrict__ scale, const float lr);

__global__ void Channel_Squeeze1D(const float* __restrict__ X, float*  __restrict__ average, const int batch, const int depth, const int a, const int b);


template <typename T1, typename T2>
__global__ void SV(T1* __restrict__ X, const T2 scale, const long long total, const int type)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const T1 factor = scale;
    if(type==0) X[global_idx] *= factor;
    else X[global_idx] /= factor;
}

template <typename T1, typename T2>
__global__ void SG(const T1* __restrict__ X, T1* __restrict__ Y, const T2 scale, const long long total, const int deriv)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const T1 factor = scale;
    if(deriv==0) Y[global_idx] = X[global_idx] * factor;
    else Y[global_idx] += X[global_idx] * factor;

}

template <typename T1, typename T2>
__global__ void SP(T1* __restrict__ X, const T2 *scale, const long long total, const int type)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const T1 factor = scale[0];
    if(type==0) X[global_idx] *= factor;
    else X[global_idx] /= factor;
}


__global__ void Sqrt_Scale(double* __restrict__ X, const double scale, const int type = 0);


__global__ void mse_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total, const bool last, const bool deriv);

__global__ void  ce_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total, const bool last, const bool deriv);

__global__ void   e_kernel(const float* __restrict__ X, const float* __restrict__ grad, float* __restrict__ output, const long long total, const bool last, const bool deriv);


__global__ void scalar_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, float*__restrict__ output, const long long total);

__global__ void scalar_ce_kernel(const  float* __restrict__ X, const float* __restrict__ target, float*__restrict__ output, const long long total);

__global__ void scalar_e_kernel( const  float* __restrict__ X, float* __restrict__ output, const int batch, const long long total);


__global__ void deriv_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total,const bool last, const bool full);

__global__ void deriv_ce_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total, const bool last, const bool full);

__global__ void deriv_e_kernel(const float* __restrict__ X, const float* __restrict__ grad, float* __restrict__ output, const int batch, const long long total, const bool last, const bool full);

__global__ void idx_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ t_idx, float* __restrict__ out, const int batch, const int col);

__global__ void idx_mse_backward(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ t_idx, const float* __restrict__ grad, float* __restrict__ out, const int batch, const int col, const bool last, const bool full);

__global__ void BatchMinMaxNorm(float* __restrict__ X, const float* __restrict__ max, const float *__restrict__ min, const int batch, const long long total_size);

__global__ void BatchMinMaxDeNorm(float* __restrict__ X, const float max, const float min,const int batch, const long long total_size);

__global__ void StdNorm(float* __restrict__ X, const float img_max, const float mean, const float std, const long long total);

__global__ void StdDeNorm(float* __restrict__ X, const float img_max, const float mean, const float std, const long long total);

__global__ void AddNoise(float* __restrict__ X, float* __restrict__ noise, const int t, const int T, const long long total_size);

__global__ void UnifNoise(float*__restrict__ Y, const long long total_size, const uint64_t seed);

__global__ void GaussianNoise(float* __restrict__ Y, const long long total_size, const uint64_t seed);

__global__ void ReplaceNoise(float* __restrict__ X, const float* __restrict__ Y, const float beta, const long long total_size, const uint64_t seed);

__global__ void BMax(const float* __restrict__ data, float* __restrict__ value, int batch, int channel, int out_size);

__global__ void BMin(const float* __restrict__ data, float* __restrict__ value, int batch, int channel, int out_size);

__global__ void FindMax(const float* __restrict__ data, float* __restrict__ maxArr, int batch, int channels, int row, int col, int type);


__global__ void SumRows(const float* __restrict__ data, float* __restrict__ arr, const int batch, const int channels, const int row, const int col);

__global__ void SumCols(const float* __restrict__ data, float* __restrict__ arr, const int batch, const int channels, const int row, const int col);

__global__ void Scale_arr(float* __restrict__ data, const float* __restrict__ arr, const int batch, const int channels, const int row, const int col, const int mode, const int transposed);

__global__ void natural_logarithm(const float* __restrict__ data, const float* __restrict__ surrogate_deriv, float* __restrict__ output, const long long total_size, const int deriv=0);

__global__ void exponentiate(const float* __restrict__ data, const float* __restrict__ surrogate_deriv, float* __restrict__ output, const long long total_size, const int deriv=0);
__global__ void exponentiate(const float* __restrict__ data, float* __restrict__ output, const float* __restrict__ maxArr, const int channels, const int row, const int col, const int type, const long long total_size);

void SoftMax(const float* __restrict__ data, float* __restrict__ arr, float* __restrict__ output, float* __restrict__ maxArr, const int batch, const int channels, const int row, const int col, const int type);

__global__ void deriv_softmax(const float* __restrict__ output, const float* __restrict__ grad, float* __restrict__ dinput, const int batch, const int channels, const int row, const int col, const int type);

void deriv_SoftMax(const float* __restrict__ output,const float* __restrict__ grad, float* __restrict__ dinput, const int batch,const int channels,const int row,const int col, const int type);

__global__ void exponentiateM(const float* __restrict__ data, float* __restrict__ output, const float* __restrict__ maxArr, const int channels, const int row, const int col, const int type, const long long total_size);

void SoftMask(const float*__restrict__ data, float* __restrict__ arr, float* __restrict__ output, float* __restrict__ maxArr, const int batch, const int channels, const int row, const int col, const int type);

__global__ void ISNAN(const float* __restrict__ X, const long long total);

__global__ void GatherEmbeddings(float* __restrict__ output,const float* __restrict__ EmbedSpace,const int* __restrict__ keys, int c, int max_c, const int embed_dim, const long long total);

template <typename T1, typename T2>
void ScaleValue(T1*  __restrict__ X, const T2 scale, const long long total, const int type = 0)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb - 1) / tpb;
    SV<<<bpg, tpb>>>(X, scale, total, type);
};

template <typename T1, typename T2>
void ScaleGraph(const T1* __restrict__  X, T1* __restrict__ Y, const T2 scale, const long long total, const int deriv = 0)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb - 1) / tpb;
    SG<<<bpg, tpb>>>(X, Y, scale, total, deriv);

};

template <typename T1, typename T2>
void ScalePtr(T1* __restrict__ X, const T2* __restrict__ scale, const long long total, const int type = 0)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb - 1) / tpb;
    SP<<<bpg, tpb>>>(X, scale, total, type);
};

__global__ void TopKSampleKernel(const float* __restrict__ arr, int* __restrict__ out_idx, const int size, const int k, const float rand_val);

__global__ void ArgMax(const float* __restrict__ arr, int* __restrict__ X, const int size);

__global__ void identityKernel(float* __restrict__ input, const int batch, const int channels, const int row_col);

__global__ void clampKernel(const float* __restrict__ input, const float* __restrict__ grad, float* __restrict__ output, const float min, const float max, const long long total, const int deriv = 0);

__global__ void minKernel(const float* __restrict__ inp_A, const float* __restrict__ inp_B, float* __restrict__ mask, 
                          float* __restrict__ grad_A, float* __restrict__ grad_B, float* __restrict__ output, const long long size, const int deriv=0);

__global__ void maxKernel(const float* __restrict__ inp_A, const float* __restrict__ inp_B, float* __restrict__ mask, 
                          float* __restrict__ grad_A, float* __restrict__ grad_B, float* __restrict__ output, const long long size, const int deriv=0);

__global__ void getactionKernel(const float* __restrict__ input, const float* __restrict__ actions, float* __restrict__ output, const int action_dim, const long long total, const bool deriv = false);

__global__ void static_assign_values( const float* __restrict__ traj, const float* __restrict__ advantages, const float* __restrict__ returns, 
    const float* __restrict__ states, const int*   __restrict__ idx,  
    float* __restrict__ out_actions, float* __restrict__ out_logp, float* __restrict__ out_adv,
    float* __restrict__ out_ret, float* __restrict__ out_states,
    const int total, const int state_dim, const int mini_batch_offset);


__global__ void dqn_assign_values(const float* __restrict__ state_replay, const float* __restrict__ replay_traj, 
    float* __restrict__ state, float* __restrict__ target, float* __restrict__ target_indices,
    const int* __restrict__ d_idx, const int state_total, const int batch, const int start);
