#include "kernels.h"

__device__ __forceinline__ double warpReduceSum(double val, int offset)
{
    for (int off = offset; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ double block_reduce_sum(double val, double* smem)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int nWarps = (blockDim.x + 31) >> 5;

    val = warpReduceSum(val, 16);      

    if (lane == 0) smem[warpId] = val;
    __syncthreads();

    val = 0.0;
    if (threadIdx.x == 0)
        for (int w = 0; w < nWarps; ++w) val += smem[w];

    return val;                           
}

__device__ double atomicAddDouble(double* __restrict__ address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ inline void atomicMaxFloat(float* __restrict__ addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ inline void atomicMinFloat(float* __restrict__ addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ int ceil_div(const int a, const int b){return (a + b - 1) / b;}

__global__ void permute(const float* __restrict__ X, float* __restrict__ Y, const int d0, const int d1, const int d2, const int d3, const int i0, const int i1, const int i2, const int i3)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= d0*d1*d2*d3) return;
    
    int dims[4] = {d0, d1, d2, d3};
    int perm[4] = {i0, i1, i2, i3};
    int orig_coords[4];
    long long temp = idx;

    orig_coords[3] = temp % d3; temp /= d3;
    orig_coords[2] = temp % d2; temp /= d2;
    orig_coords[1] = temp % d1; temp /= d1;
    orig_coords[0] = temp;
    

    int new_coords[4];
    new_coords[0] = orig_coords[perm[0]];
    new_coords[1] = orig_coords[perm[1]];
    new_coords[2] = orig_coords[perm[2]];
    new_coords[3] = orig_coords[perm[3]];
    
    int new_dims[4] = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};

    long long out_idx = new_coords[0];
    out_idx = out_idx * new_dims[1] + new_coords[1];
    out_idx = out_idx * new_dims[2] + new_coords[2];
    out_idx = out_idx * new_dims[3] + new_coords[3];
    Y[out_idx] = X[idx];
}

__global__ void mulKernel(const float*  X, const float* Y, float* Z, const long long total_size, const bool deriv)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= total_size) return;
    if(deriv) Z[idx] += X[idx]*Y[idx];
    else Z[idx] = X[idx]*Y[idx];
}

__global__ void PEncoding(float* __restrict__ X, const int t, const int t_dim, const int total_size)
{
    const int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total_size) return;
    
    const int half_dim = t_dim / 2;
    const int freq_idx = (global_idx < half_dim) ? global_idx : global_idx - half_dim;
    
    const double exponent = -(double)freq_idx * log(10000.0) / (half_dim - 1);
    const double freq = exp(exponent);
    const double angle = (double)t * freq;

    if(global_idx < half_dim) {
        X[global_idx] = (float)cos(angle);
    } else {
        X[global_idx] = (float)sin(angle);
    }
}

__global__ void MatPEncoding(const float* __restrict__ X, float* __restrict__ output, const int batch,
                             const int row, const int col, const int total, const int start_idx)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total) return;
    
    const int pos_idx = (global_idx % (row * col)) / col  + start_idx;
    const int dim_idx = global_idx % col;
    const int i = dim_idx / 2;

    const double exponent = (2.0 * i) / col;
    const double freq = 1.0 / pow(10000.0, exponent);
    const double angle = (double)pos_idx * freq;

    if(dim_idx % 2 == 0) {
        output[global_idx] = X[global_idx] + (float)sin(angle);
    } else {
        output[global_idx] = X[global_idx] + (float)cos(angle);
    }

}


__global__ void bmm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                    const int batch_size, const int m, const int n, const int p, const int backward, 
                    const int A, const int B, const int C, const float scale) //has atomic
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    int batch_offset_a = (A == 0)? 0 : batch_idx * m * n;
    int batch_offset_b = (B == 0)? 0 : batch_idx * n * p;
    int batch_offset_c = (C == 0)? 0 : batch_idx * m * p;
    
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < m && (t * BLOCK_SIZE + threadIdx.x) < n)
            s_A[threadIdx.y][threadIdx.x] = a[batch_offset_a + (row * n) + t * BLOCK_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < p && (t * BLOCK_SIZE + threadIdx.y) < n)
            s_B[threadIdx.y][threadIdx.x] = b[batch_offset_b + (t * BLOCK_SIZE + threadIdx.y) * p + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    if(row < m && col < p)
    {
    if(backward == 0) c[batch_offset_c + row * p + col] = scale *sum;
    else c[batch_offset_c + row * p + col] += scale * sum;
    }

}

__global__ void bmmABT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                       const int batch_size, const int m, const int n, const int p, const int backward, 
                       const int A, const int B, const int C, const float scale) //has atomic 
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Calculate batch offsets for contiguous memory layout
    int batch_offset_a = (A == 0)? 0 : batch_idx * m * n;
    int batch_offset_b = (B == 0)? 0 : batch_idx * p * n;  // B^T is stored as p × n
    int batch_offset_c = (C == 0)? 0 : batch_idx * m * p;
    
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < m && (t * BLOCK_SIZE + threadIdx.x) < n)
            s_A[threadIdx.y][threadIdx.x] = a[batch_offset_a + row * n + t*BLOCK_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < p && (t * BLOCK_SIZE + threadIdx.y) < n)
            s_B[threadIdx.y][threadIdx.x] = b[batch_offset_b + col * n + t*BLOCK_SIZE + threadIdx.y]; // B^T access
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        
        __syncthreads();
    }
    if(row < m && col < p)
    {
    if(backward ==0) c[batch_offset_c + row * p + col] = scale * sum;
    else c[batch_offset_c + row * p + col] += scale * sum;
    }

    
}

__global__ void bmmATB(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                       const int batch_size, const int m, const int n, const int p, const int backward, 
                       const int A, const int B, const int C, const float scale) 
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    int row = threadIdx.y + blockIdx.y * blockDim.y; 
    int col = threadIdx.x + blockIdx.x * blockDim.x; 

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE]; 
    float sum = 0.0f;

    int batch_offset_a = (A==0)? 0: batch_idx*m*n; // A [m × n]
    int batch_offset_b = (B==0)? 0: batch_idx*m*p; // B [m × p]
    int batch_offset_c = (C==0)? 0: batch_idx*n*p; // C [n × p]
    
    for (int t = 0; t < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        
        int k_A = t * BLOCK_SIZE + threadIdx.x;
        if (k_A < m && row < n) { s_A[threadIdx.y][threadIdx.x] = a[batch_offset_a + k_A * n + row];} 
        else {s_A[threadIdx.y][threadIdx.x] = 0.0f;}
        
        int k_B = t * BLOCK_SIZE + threadIdx.y;
        if (k_B < m && col < p) {
            s_B[threadIdx.y][threadIdx.x] = b[batch_offset_b + k_B * p + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    if (row < n && col < p)
    {
    if (backward == 0) c[batch_offset_c + row * p + col] = scale * sum;
    else c[batch_offset_c + row * p + col] += scale * sum;
    }

}

__global__ void bmmATBT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                        const int batch_size, const int m, const int n, const int p, const int backward, 
                        const int A, const int B, const int C, const float scale)
{
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    int row = threadIdx.y + blockIdx.y * blockDim.y;  // C row → n
    int col = threadIdx.x + blockIdx.x * blockDim.x;  // C col → p

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    int batch_offset_a = (A==0) ? 0 : batch_idx * m * n; // A [m×n]
    int batch_offset_b = (B==0) ? 0 : batch_idx * p * m; // B [p×m] (for Bᵀ)
    int batch_offset_c = (C==0) ? 0 : batch_idx * n * p; // C [n×p]

    for (int t = 0; t < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

        int k_A = t * BLOCK_SIZE + threadIdx.x;
        if (k_A < m && row < n)
            s_A[threadIdx.y][threadIdx.x] = a[batch_offset_a + k_A * n + row];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        int k_B = t * BLOCK_SIZE + threadIdx.y;
        if (k_B < m && col < p)
            s_B[threadIdx.y][threadIdx.x] =
                b[batch_offset_b + col * m + k_B];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < p) {
        if (!backward) c[batch_offset_c + row * p + col] = scale * sum;
        else c[batch_offset_c + row * p + col] += scale * sum;
    }
}

__device__ __forceinline__ int batch_chan_offset(int flag, int batch_idx, int channel_idx, int num_channels, int inner_size)
{
    /*
    0 (bits 00) → [1 × 1 × …] broadcast both
    1 (bits 01) → [B × 1 × …] has batch, broadcast channel
    2 (bits 10) → [1 × C × …] broadcast batch, has channel
    3 (bits 11) → [B × C × …] has both
    */

    const int has_batch = flag & 1;
    const int has_chan  = (flag >> 1) & 1;
    const int chan_stride_in_batch = has_chan ? num_channels : 1;
    return (has_batch ? batch_idx * chan_stride_in_batch : 0) * inner_size
         + (has_chan  ? channel_idx                      : 0) * inner_size;
}

__global__ void bcmm(const float* __restrict__ a, const float* __restrict__ b,float* __restrict__ c, 
                    const  int batch_size, const int num_channels, const int m, const int n, const int p,
                    const int backward, const int A, const int B, const int C, const float scale)
{
        /*
    @brief: 0 (bits 00) → [1 × 1 × …] broadcast both
            1 (bits 01) → [B × 1 × …] has batch, broadcast channel
            2 (bits 10) → [1 × C × …] broadcast batch, has channel
            3 (bits 11) → [B × C × …] has both
    */


    const int linear_idx = blockIdx.z;
    if (linear_idx >= batch_size * num_channels) return;
    const int batch_idx   = linear_idx / num_channels;
    const int channel_idx = linear_idx % num_channels;

    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    const int off_a = batch_chan_offset(A, batch_idx, channel_idx, num_channels, m * n);
    const int off_b = batch_chan_offset(B, batch_idx, channel_idx, num_channels, n * p);
    const int off_c = batch_chan_offset(C, batch_idx, channel_idx, num_channels, m * p);

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        s_A[threadIdx.y][threadIdx.x] =
            (row < m && (t * BLOCK_SIZE + threadIdx.x) < n)
            ? a[off_a + row * n + t * BLOCK_SIZE + threadIdx.x]
            : 0.0f;

        s_B[threadIdx.y][threadIdx.x] =
            (col < p && (t * BLOCK_SIZE + threadIdx.y) < n)
            ? b[off_b + (t * BLOCK_SIZE + threadIdx.y) * p + col]
            : 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < p) {
        if (backward == 0) c[off_c + row * p + col]  = scale * sum;
        else               c[off_c + row * p + col] += scale * sum;
    }
}

__global__ void bcmmABT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                       const int batch_size, const int num_channels, const int m, const int n, const int p,
                       const int backward, const int A, const int B, const int C, const float scale)
{
        /*
    @brief: 0 (bits 00) → [1 × 1 × …] broadcast both
            1 (bits 01) → [B × 1 × …] has batch, broadcast channel
            2 (bits 10) → [1 × C × …] broadcast batch, has channel
            3 (bits 11) → [B × C × …] has both
    */

    const int linear_idx = blockIdx.z;
    if (linear_idx >= batch_size * num_channels) return;
    const int batch_idx   = linear_idx / num_channels;
    const int channel_idx = linear_idx % num_channels;

    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    const int off_a = batch_chan_offset(A, batch_idx, channel_idx, num_channels, m * n);
    const int off_b = batch_chan_offset(B, batch_idx, channel_idx, num_channels, p * n);
    const int off_c = batch_chan_offset(C, batch_idx, channel_idx, num_channels, m * p);

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        s_A[threadIdx.y][threadIdx.x] =
            (row < m && (t * BLOCK_SIZE + threadIdx.x) < n)
            ? a[off_a + row * n + t * BLOCK_SIZE + threadIdx.x]
            : 0.0f;

        s_B[threadIdx.y][threadIdx.x] =
            (col < p && (t * BLOCK_SIZE + threadIdx.y) < n)
            ? b[off_b + col * n + t * BLOCK_SIZE + threadIdx.y]
            : 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < p) {
        if (backward == 0) c[off_c + row * p + col]  = scale * sum;
        else               c[off_c + row * p + col] += scale * sum;
    }
}

__global__ void bcmmATB(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                       const int batch_size, const int num_channels, const int m, const int n, const int p,
                       const int backward, const int A, const int B, const int C, const float scale)
{

            /*
    @brief: 0 (bits 00) → [1 × 1 × …] broadcast both
            1 (bits 01) → [B × 1 × …] has batch, broadcast channel
            2 (bits 10) → [1 × C × …] broadcast batch, has channel
            3 (bits 11) → [B × C × …] has both
    */

    const int linear_idx = blockIdx.z;
    if (linear_idx >= batch_size * num_channels) return;
    const int batch_idx   = linear_idx / num_channels;
    const int channel_idx = linear_idx % num_channels;

    const int row = threadIdx.y + blockIdx.y * blockDim.y;  // → n
    const int col = threadIdx.x + blockIdx.x * blockDim.x;  // → p

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    const int off_a = batch_chan_offset(A, batch_idx, channel_idx, num_channels, m * n);
    const int off_b = batch_chan_offset(B, batch_idx, channel_idx, num_channels, m * p);
    const int off_c = batch_chan_offset(C, batch_idx, channel_idx, num_channels, n * p);

    for (int t = 0; t < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Aᵀ access: A[k, row]
        const int k_A = t * BLOCK_SIZE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] =
            (k_A < m && row < n)
            ? a[off_a + k_A * n + row]
            : 0.0f;

        const int k_B = t * BLOCK_SIZE + threadIdx.y;
        s_B[threadIdx.y][threadIdx.x] =
            (k_B < m && col < p)
            ? b[off_b + k_B * p + col]
            : 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < p) {
        if (backward == 0) c[off_c + row * p + col]  = scale * sum;
        else               c[off_c + row * p + col] += scale * sum;
    }
}

__global__ void bcmmATBT(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                        const int batch_size, const int num_channels, const int m, const int n, const int p,
                        const int backward,   const int A, const int B, const int C, const float scale)
{
            /*
    @brief: 0 (bits 00) → [1 × 1 × …] broadcast both
            1 (bits 01) → [B × 1 × …] has batch, broadcast channel
            2 (bits 10) → [1 × C × …] broadcast batch, has channel
            3 (bits 11) → [B × C × …] has both
    */

    const int linear_idx = blockIdx.z;
    if (linear_idx >= batch_size * num_channels) return;
    const int batch_idx   = linear_idx / num_channels;
    const int channel_idx = linear_idx % num_channels;

    const int row = threadIdx.y + blockIdx.y * blockDim.y;  
    const int col = threadIdx.x + blockIdx.x * blockDim.x; 

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    const int off_a = batch_chan_offset(A, batch_idx, channel_idx, num_channels, m * n);
    const int off_b = batch_chan_offset(B, batch_idx, channel_idx, num_channels, p * m);
    const int off_c = batch_chan_offset(C, batch_idx, channel_idx, num_channels, n * p);

    for (int t = 0; t < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        const int k_A = t * BLOCK_SIZE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] =
            (k_A < m && row < n)
            ? a[off_a + k_A * n + row]
            : 0.0f;

        const int k_B = t * BLOCK_SIZE + threadIdx.y;
        s_B[threadIdx.y][threadIdx.x] =
            (k_B < m && col < p)
            ? b[off_b + col * m + k_B]
            : 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < p) {
        if (!backward) c[off_c + row * p + col]  = scale * sum;
        else           c[off_c + row * p + col] += scale * sum;
    }
}


__global__ void LayerMean(const float* __restrict__ data, float* __restrict__ mean,
                          const int batch, const int channels, const int row, const int col, const bool scale)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int batch_idx  = blockIdx.x;         
    const int out_size   = row * col;
    const int group_size = channels * out_size;  
    const int tid        = threadIdx.x;

    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int channel_idx = i / out_size;
        const int spatial     = i % out_size;
        sum += data[batch_idx * channels * out_size + channel_idx * out_size + spatial];
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        if(scale == true) mean[batch_idx] = smem[0] / group_size;
        else mean[batch_idx] = smem[0];
    } 
}

__global__ void BatchMean(const float* __restrict__ data, float* __restrict__ mean, 
                          const int batch, const int channels, const int row, const int col, const bool scale)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int channel_idx = blockIdx.x;           // one block per channel
    const int out_size    = row * col;
    const int group_size  = batch * out_size;     // elements this block reduces over
    const int tid         = threadIdx.x;

    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int batch_idx = i / out_size;
        const int spatial   = i % out_size;
        sum += data[batch_idx * channels * out_size + channel_idx * out_size + spatial];
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        if(scale == true) mean[channel_idx] = smem[0] / group_size;
        else mean[channel_idx] = smem[0];
    } 
}

__global__ void GroupMean(const float* __restrict__ data, float* __restrict__ mean, const int batch, 
                          const int channels, const int groups, const int row, const int col, const bool scale)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int group_size = (channels / groups) * row * col;  
    const int batch_idx  = blockIdx.x / groups;
    const int group_idx  = blockIdx.x % groups;
    const int ch_start   = group_idx * (channels / groups);
    const int tid        = threadIdx.x;
    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int local_ch  = i / (row * col);
        const int spatial   = i % (row * col);
        const int ch        = ch_start + local_ch;
        sum += data[(batch_idx * channels + ch) * (row * col) + spatial];
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) 
    {
        if(scale == true) mean[blockIdx.x] = smem[0] / group_size;
        else mean[blockIdx.x] = smem[0];
    } 
}

__global__ void InstanceMean(const float* __restrict__ data, float* __restrict__ mean,
                             const int batch,const int channels,const int row, const int col, const bool scale)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int out_idx  = blockIdx.x;             
    const int out_size = row * col;
    const int tid      = threadIdx.x;

    float sum = 0.0;
    for (int i = tid; i < out_size; i += blockDim.x)
        sum += data[out_idx * out_size + i];

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        if(scale == true) mean[out_idx] = smem[0] / out_size;
        else mean[out_idx] = smem[0];
    } 
}


__global__ void LayerStd(const float* __restrict__ data, const float* __restrict__ mean, float* __restrict__ std,
                         const int batch, const int channels, const int row, const int col)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int batch_idx  = blockIdx.x;
    const int out_size   = row * col;
    const int group_size = channels * out_size;
    const int tid        = threadIdx.x;
    const float mu      = mean[batch_idx];

    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int channel_idx = i / out_size;
        const int spatial     = i % out_size;
        const float diff     = data[batch_idx * channels * out_size + channel_idx * out_size + spatial] - mu;
        sum += diff * diff;
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) std[batch_idx] = smem[0];
}

__global__ void BatchStd(const float* __restrict__ data, const float* __restrict__ mean, float* __restrict__ std,
                         const int batch, const int channels, const int row, const int col)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int channel_idx = blockIdx.x;
    const int out_size    = row * col;
    const int group_size  = batch * out_size;
    const int tid         = threadIdx.x;
    const float mu       = mean[channel_idx];

    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int batch_idx = i / out_size;
        const int spatial   = i % out_size;
        const float diff   = data[batch_idx * channels * out_size + channel_idx * out_size + spatial] - mu;
        sum += diff * diff;
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) std[channel_idx] = smem[0];
}

__global__ void GroupStd(const float* __restrict__ data, const float* __restrict__ mean, float* __restrict__ var,
                         const int batch, const int channels, const int groups, const int row, const int col)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int group_size = (channels / groups) * row * col;
    const int batch_idx  = blockIdx.x / groups;
    const int group_idx  = blockIdx.x % groups;
    const int ch_start   = group_idx * (channels / groups);
    const int tid        = threadIdx.x;
    const float mu      = mean[blockIdx.x];

    float sum = 0.0;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int local_ch = i / (row * col);
        const int spatial  = i % (row * col);
        const int ch       = ch_start + local_ch;
        const float diff  = data[(batch_idx * channels + ch) * (row * col) + spatial] - mu;
        sum += diff * diff;
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) var[blockIdx.x] = smem[0]; 
}

__global__ void InstanceStd(const float* __restrict__ data, const float* __restrict__ mean, float* __restrict__ std,
                            const int batch, const int channels, const int row, const int col)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int out_idx  = blockIdx.x;
    const int out_size = row * col;
    const int tid      = threadIdx.x;
    const float mu    = mean[out_idx];

    float sum = 0.0;
    for (int i = tid; i < out_size; i += blockDim.x)
    {
        const float diff = data[out_idx * out_size + i] - mu;
        sum += diff * diff;
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) std[out_idx] = smem[0];
}


__global__ void LNorm(float* __restrict__ data, const float* __restrict__ mean, const float*  __restrict__ std,
                      const int batch,const int channels, const int row,const int col,const float gamma,
                      const float beta,const float epsilon) //γ*((x-μ)/sqrtf(σ^2+ε))+β;
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size = row*col;
    const int batch_idx   = (global_idx / (channels*out_size));
    data[global_idx] = gamma*((data[global_idx]-mean[batch_idx]) / 
    sqrtf((1.0f/(channels*out_size))*std[batch_idx] + epsilon)) + beta;

}

__global__ void BNorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ std,
                      const int batch, const int channels, const int row, const int col, const float gamma,
                      const float beta, const float epsilon) //γ*((x-μ)/sqrtf(σ^2+ε))+β;
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size  = row*col;
    const int channel_idx = (global_idx / out_size) % channels;
    data[global_idx] = gamma*((data[global_idx]-mean[channel_idx]) / 
    sqrtf((1.0f/(batch*out_size))*std[channel_idx] + epsilon))+beta;
    
}

__global__ void GNorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__ var,
                      const int batch, const int channels, const int groups, const int row, const int col,
                      const float gamma, const float beta, const float epsilon)  //γ*((x-μ)/sqrtf(σ^2+ε))+β;
{
    const long long global_idx = threadIdx.x + (long long)blockDim.x * blockIdx.x;
    if (global_idx >= (long long)batch * channels * row * col) return;

    const int out_size    = row * col;
    const int depth       = channels / groups;
    const int batch_idx   = (int)(global_idx / (channels * out_size));
    const int channel_idx = (int)(global_idx / out_size) % channels;
    const int group_idx   = channel_idx / depth;
    const int g           = batch_idx * groups + group_idx;

    const float inv_N = 1.0f / (float)(depth * out_size);
    const float mu    = (float)mean[g];
    const float sigma = sqrtf((float)var[g] * inv_N + epsilon); 

    data[global_idx] = gamma * (data[global_idx] - mu) / sigma + beta;
}

__global__ void INorm(float* __restrict__ data, const float* __restrict__ mean, const float* __restrict__  std,
                      const int batch, const int channels, const int row, const int col, const float gamma, 
                      const float beta, const float epsilon) //γ*((x-μ)/sqrtf(σ^2+ε))+β;
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size = row*col;
    const int batch_idx   = (global_idx / (channels*out_size));
    const int channel_idx = (global_idx / out_size) % channels;
    data[global_idx] = gamma*((data[global_idx] - mean[batch_idx*channels+channel_idx]) 
                       / sqrtf((1.0f/out_size)*std[batch_idx*channels+channel_idx] + epsilon)) + beta;
}


__global__ void LayerBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, 
                              const float* __restrict__ ggamma, const float* __restrict__ ggammanode, const float* __restrict__ variance, 
                              const float gamma, const float epsilon, const int batch, const int channels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*row*col) return;
    const int out_size = row*col;
    const int batch_idx = (global_idx / out_size);
    const int N = channels * out_size; 
    
    const float std_inv = 1.0f / sqrtf(variance[batch_idx] / N + epsilon);
    igrad[global_idx] += std_inv * (gamma * ngrad[global_idx]-(ggamma[batch_idx]/N) - node[global_idx] * (ggammanode[batch_idx] / N));

}

__global__ void BatchBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, 
                              const float* __restrict__ ggamma, const float* __restrict__ ggammanode, const float* __restrict__ variance, 
                              const float gamma, const float epsilon, const int batch, const int channels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size  = row*col;
    const int channel_idx = (global_idx / out_size) % channels;
    const int N = batch * out_size; 
    const float std_inv = 1.0f/sqrtf(variance[channel_idx]/ N + epsilon);
    igrad[global_idx] += std_inv * (gamma * ngrad[global_idx] - (ggamma[channel_idx] / N) - node[global_idx] * (ggammanode[channel_idx] / N));
}

__global__ void GroupBackward(float*__restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma,const float* __restrict__ ggammanode, 
                              const float* __restrict__ variance, const float gamma, const float epsilon, const int batch, 
                              const int channels, const int groups, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= batch*channels*row*col) return;
    const int out_size = row*col;
    const int batch_idx   = (global_idx / (channels*out_size));
    const int channel_idx = (global_idx / out_size) % channels;
    const int group_idx   = channel_idx / (channels / groups);
    const int depth = channels / groups;
    const int N = depth * out_size;  
    const int variance_idx = batch_idx * groups + group_idx;
    const float std_inv = 1.0f / sqrtf(variance[variance_idx] / N + epsilon);
    
    igrad[global_idx] += std_inv * (gamma * ngrad[global_idx] - (ggamma[variance_idx] / N)
                         - node[global_idx] * (ggammanode[variance_idx] / N));


}

__global__ void InstanceBackward(float* __restrict__ igrad, const float* __restrict__ node, const float* __restrict__ ngrad, const float* __restrict__ ggamma,const float* __restrict__ ggammanode, 
                              const float* __restrict__ variance, const float gamma, const float epsilon, const int batch, 
                              const int channels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size = row*col;
    const int batch_idx   = (global_idx / (channels*out_size));
    const int channel_idx = (global_idx / out_size) % channels;
    const int N = out_size; 
    const int variance_idx = batch_idx * channels + channel_idx;
    const float std_inv = 1.0f / sqrtf(variance[variance_idx] / N + epsilon);
    
    igrad[global_idx] += std_inv * (gamma * ngrad[global_idx] - (ggamma[variance_idx] / N)
                         - node[global_idx] * (ggammanode[variance_idx] / N));

}


__global__ void LayerMeanGrad(const float* __restrict__ ngrad, float* __restrict__ igrad, const int batch, const int channels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*row*col) return;
    const int out_size = row*col;
    const int N = channels * out_size; 
    const int batch_idx = global_idx / N;
    igrad[global_idx] += ngrad[batch_idx] / N;
}

__global__ void BatchMeanGrad(const float* __restrict__ ngrad, float* __restrict__ igrad, const int batch, const int channels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >=batch*channels*row*col) return;
    const int out_size = row*col;
    const int N = batch * out_size; 
    const int channel_idx = (global_idx / out_size) % channels;
    igrad[global_idx] += ngrad[channel_idx] / N;
}

__global__ void dropoutKernel(const float* __restrict__ data, float* __restrict__ mask, float* __restrict__ output, const long long size, const float p, const uint64_t seed, const int deriv)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    if(deriv == 0)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        float val = curand_uniform(&state);
        mask[idx] = (val > p) ? 1.0f : 0.0f;
        output[idx] = data[idx] * mask[idx] * (1.0f / (1.0f - p));
    }

    else {output[idx] += data[idx] * mask[idx] * (1.0f / (1.0f - p));}
}

__global__ void Standard_Weights(float* __restrict__ w, const long long size, const float scale, const uint64_t seed){
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    float val = curand_uniform(&state);
    w[idx] = (val * 2.0f - 1.0f) * scale;
}

__global__ void PadOutput(const float* __restrict__ input, float* __restrict__ output, const int batch, const int channels, 
                          const int inH, const int inW, const int outH, const int outW, const int padTop, const int padLeft)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)batch * (long long)channels *(long long)outH * (long long)outW;
    if(idx >= total) return;

    const int w = idx % outW;
    const int h = (idx / outW) % outH;
    const int c = (idx / (outW * outH)) % channels;
    const int b = idx / (outW * outH * channels);
    const int in_h = h - padTop;
    const int in_w = w - padLeft;
    if (in_h >= 0 && in_h < inH && in_w >= 0 && in_w < inW) 
    {
        int in_idx = ((b * channels + c) * inH + in_h) * inW + in_w;
        output[idx] += input[in_idx];
    }
    return;
        

}

__global__ void CV2D(const float* __restrict__ X, const float* __restrict__ K, const float* __restrict__ bias, float* __restrict__ Y, 
                     const int N, const int Cout, const int Cin, const int H, const int W, const int KH, const int KW, const int pad, const int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int n_cout = blockIdx.z * blockDim.z + threadIdx.z;

    int n = n_cout / Cout;
    int cout = n_cout % Cout;

    if (n >= N) return;

    int OH = (2 * pad + H - KH) / stride + 1;
    int OW = (2 * pad + W - KW) / stride + 1;

    if (out_x >= OW || out_y >= OH) return;

    float sum = 0.0f;

    for (int cin = 0; cin < Cin; ++cin)
    for (int kh = 0; kh < KH; ++kh)
    for (int kw = 0; kw < KW; ++kw) {
        int in_y = out_y * stride - pad + kh;
        int in_x = out_x * stride - pad + kw;
        
        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
            long long x_id = ((long long)n * Cin + cin) * (H * W) + in_y * W + in_x;
            long long k_id = ((long long)cout * Cin + cin) * (KH * KW) + kh * KW + kw;
            sum += X[x_id] * K[k_id];
        }
    }

    long long y_id = ((long long)n * Cout + cout) * (OH * OW) + out_y * OW + out_x;
    Y[y_id] = (bias != nullptr) ? sum + bias[cout] : sum;
}

__global__ void GV2D(const float* __restrict__ X, const float* __restrict__ dZ, float* __restrict__ dK, const int batch, 
                     const int out,const int in,const int a,const int b,const int c,const int d,const int pad,const int stride) // Assignment gradient kernel
{
    int cout = blockIdx.x * blockDim.x + threadIdx.x;
    int cin  = blockIdx.y * blockDim.y + threadIdx.y;
    int kpos = blockIdx.z * blockDim.z + threadIdx.z;
    int kh = kpos / d;
    int kw = kpos % d;
    if (cout >= out || cin >= in || kh >= c || kw >= d) return;

    float sum = 0.0f;

    int OH =  (2 * pad + a - c) / stride + 1;                   
    int OW =  (2 * pad + b - d) / stride + 1;

    for (int n = 0; n < batch; ++n)
    for (int oy = 0; oy < OH; ++oy)
    for (int ox = 0; ox < OW; ++ox) 
    {
        int in_y = oy * stride - pad + kh;
        int in_x = ox * stride - pad + kw;
        if (in_y < 0 || in_y >= a || in_x < 0 || in_x >= b) continue;

        long long x_id = ((long long)n * in + cin) * (a * b) + in_y * b + in_x;
        long long g_id = ((long long)n * out + cout) * (OH * OW) + oy * OW + ox;
        sum += X[x_id] * dZ[g_id];
    }

    long long k_id = ((long long)cout * in + cin) * (c * d)+ kh * d + kw;
    dK[k_id] = sum;
}

__global__ void CV2D_GradInput(const float* __restrict__ dY,const float* __restrict__ K, float* __restrict__ dX,
                               const int batch, const int out, const int inp, const int a, const int b,const int c,
                               const int d, const int outR, const int outC, const int pad,  const int stride)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;

    if (iw >= b || ih >= a || z >= batch * inp) return;

    const int n  = z / inp;
    const int ic = z % inp;

    float sum = 0.0f;

    for (int oc = 0; oc < out; oc++) {
        for (int kh = 0; kh < c; kh++) {
            int num_r = ih + pad - kh;
            if (num_r < 0 || num_r % stride != 0) continue;
            int oy = num_r / stride;
            if (oy >= outR) continue;

            for (int kw = 0; kw < d; kw++) {
                int num_c = iw + pad - kw;
                if (num_c < 0 || num_c % stride != 0) continue;
                int ox = num_c / stride;
                if (ox >= outC) continue;

                int dy_idx = ((n * out + oc) * outR + oy) * outC + ox;
                int k_idx  = ((oc * inp + ic) * c + kh) * d + kw;
                sum += dY[dy_idx] * K[k_idx];
            }
        }
    }

    dX[((n * inp + ic) * a + ih) * b + iw] += sum;
}

__global__ void CVT2D(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, 
                      float* __restrict__ output, const int batch, const int out_channels,  const int inp_channels,  
                      const int inp_h, const int inp_w, const int kernel_h, const int kernel_w, const int pad, const int stride)
{
    const int out_h = (inp_h - 1) * stride - 2 * pad + kernel_h;
    const int out_w = (inp_w - 1) * stride - 2 * pad + kernel_w;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (col >= out_w || row >= out_h || z >= batch * out_channels) return;
    
    int b = z / out_channels;
    int oc = z % out_channels;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < inp_channels; ic++) {
    for (int kh = 0; kh < kernel_h; kh++) {
    for (int kw = 0; kw < kernel_w; kw++) {
        int in_row = (row + pad - kh);
        int in_col = (col + pad - kw);
        if (in_row % stride == 0 && in_col % stride == 0) 
        {
            in_row /= stride;
            in_col /= stride;
                    
            if (in_row >= 0 && in_row < inp_h && in_col >= 0 && in_col < inp_w) {
                        int inp_idx = ((b * inp_channels + ic) * inp_h + in_row) * inp_w + in_col;
                        int weight_idx = ((oc * inp_channels + ic) * kernel_h + kh) * kernel_w + kw;
                        sum += input[inp_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    int out_idx = ((b * out_channels + oc) * out_h + row) * out_w + col;
    output[out_idx] = sum;
}

__global__ void GVT2D(const float* __restrict__ input,const float* __restrict__ grad_output,float* __restrict__ grad_weights,
                      const int batch, const int out_channels, const int inp_channels,const int inp_h, const int inp_w,
                      const int kernel_h, const int kernel_w,const int pad, const int stride)
{
    const int out_h = (inp_h - 1) * stride - 2 * pad + kernel_h;
    const int out_w = (inp_w - 1) * stride - 2 * pad + kernel_w;
    const int oc  = blockIdx.x;
    const int tid = threadIdx.x;
    const int inner = inp_channels * kernel_h * kernel_w;
    const int spatial = batch * inp_h * inp_w;

    for (int w = tid; w < inner; w += blockDim.x)
    {
        const int ic = w / (kernel_h * kernel_w);
        const int kh = (w % (kernel_h * kernel_w)) / kernel_w;
        const int kw = w % kernel_w;

        float sum = 0.0f;

        for (int i = 0; i < spatial; i++)
        {
            const int b      = i / (inp_h * inp_w);
            const int in_row = (i % (inp_h * inp_w)) / inp_w;
            const int in_col = i % inp_w;

            const int out_row = in_row * stride - pad + kh;
            const int out_col = in_col * stride - pad + kw;

            if (out_row >= 0 && out_row < out_h && out_col >= 0 && out_col < out_w)
            {
                const int inp_idx  = ((b * inp_channels + ic) * inp_h + in_row) * inp_w + in_col;
                const int grad_idx = ((b * out_channels + oc) * out_h + out_row) * out_w + out_col;
                sum += input[inp_idx] * grad_output[grad_idx];
            }
        }

        const int weight_idx = ((oc * inp_channels + ic) * kernel_h + kh) * kernel_w + kw;
        grad_weights[weight_idx] = sum;
    }
}

__global__ void CVT2D_GradInput(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input, 
                                const int batch, const  int out_channels, const int inp_channels, const int inp_h, const int inp_w, 
                                const int kernel_h, const int kernel_w, const int pad, const int stride)
{
    const int out_h = (inp_h - 1) * stride - 2 * pad + kernel_h;
    const int out_w = (inp_w - 1) * stride - 2 * pad + kernel_w;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (col >= inp_w || row >= inp_h || z >= batch * inp_channels) return;
    
    int b = z / inp_channels;
    int ic = z % inp_channels;
    
    float sum = 0.0f;
    for (int oc = 0; oc < out_channels; oc++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int out_row = row * stride - pad + kh;
                int out_col = col * stride - pad + kw;
                
                if (out_row >= 0 && out_row < out_h && out_col >= 0 && out_col < out_w) {
                    int grad_idx = ((b * out_channels + oc) * out_h + out_row) * out_w + out_col;
                    int weight_idx = ((oc * inp_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += grad_output[grad_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int inp_idx = ((b * inp_channels + ic) * inp_h + row) * inp_w + col;
    grad_input[inp_idx] = sum;
}

__global__ void ReLU(const float* __restrict__ input, float* __restrict__ output, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    const float val  = input[global_idx];
    if(val < 0){output[global_idx] = 0;}
    else{output[global_idx] = val;}
} 

__global__ void deriv_ReLU(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size)
{
    const long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_size) return;
    float mask = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_out[idx] += grad_in[idx] * mask;
}

__global__ void LeakyReLU(const float* __restrict__ input, float* __restrict__ output, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    const float val  = input[global_idx];
    if(val < 0){output[global_idx] = 0.01f* input[global_idx];}
    else{output[global_idx] = val;}
} 

__global__ void deriv_LeakyReLU(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size)
{
    const long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_size) return;
    float slope = 0.01f;
    float mask = (input[idx] < 0.0f) ? slope : 1.0f;
    grad_out[idx] += grad_in[idx] * mask;
}

__global__ void SiLU(const float* __restrict__ input, float* __restrict__ output, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    const float val  = input[global_idx];
    output[global_idx] = val *(1 / (1 + expf(-val)));
} 

__global__ void deriv_SiLU(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total_size) return;
    const float val = input[global_idx];
    const float exp = expf(val);
    const float negexp = 1.0f / exp;
    const float sigma = 1.0f / (1.0f + negexp);
    const float deriv = sigma + (val / (exp + negexp + 2.0f)); 
    grad_out[global_idx] += grad_in[global_idx] * deriv;
}

__global__ void TaNH(const float* __restrict__ input, float* __restrict__ output, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total_size) return;
    const float val = input[global_idx];
    const float exp = expf(val);
    const float negexp = 1.0f / exp;
    output[global_idx] = (exp - negexp) / (exp + negexp);

}

__global__ void deriv_TaNH(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total_size) return;
    const float exp2 = expf(2.0 * input[global_idx]);
    const float negexp2 = 1.0 / exp2;
    const float deriv = 4.0 / (exp2 + negexp2 + 2.0); 
    grad_out[global_idx] += grad_in[global_idx] * deriv;

}

__global__ void Sigmoid(const float* __restrict__ input, float* __restrict__ output, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total_size) return;
    const float negexp = expf( -1.0 * input[global_idx]);
    output[global_idx] = 1.0 / (1.0 + negexp);

}

__global__ void deriv_Sigmoid(const float* __restrict__ input, const float* __restrict__ grad_in, float* __restrict__ grad_out, const int total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total_size) return;
    const float exp = expf(input[global_idx]);
    const float negexp = 1.0 / exp;
    const float deriv = 1.0 / (exp + negexp + 2.0); 
    grad_out[global_idx] += grad_in[global_idx] * deriv;


}

__global__ void CopynCrop(const float* __restrict__ X, float* __restrict__ Y, const int batch_size,
                          const int depth, const int a, const int b, const int c, const int d) // (a,b) -> (c,d)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(global_idx >= batch_size * depth * c * d){return;}

    const int diff_row = (a - c) / 2;
    const int diff_col = (b - d) / 2;
    const int batch = global_idx / (depth * c * d);
    const int matrix =  (global_idx / (c * d)) % depth;
    const int temp = global_idx % (c * d); 
    const int outRow = temp / d;
    const int outCol = temp % d;
    const int xOffset = batch * depth * a * b + matrix * a * b + (outRow + diff_row) * b + outCol + diff_col;
    const int yOffset = batch * depth * c * d + matrix * c * d + outRow * d + outCol;
    Y[yOffset] = X[xOffset];

}

__global__ void PaddingCrop(const float* __restrict__ X, float* __restrict__ Y, const int batch_size,const int depth, 
                            const int a, const int b, const int c, const int d) // (c,d)->(a,b)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(global_idx >= batch_size * depth * a * b){return;}

    const int diff_row = (a - c) / 2;
    const int diff_col = (b - d) / 2;
    const int batch = global_idx / (depth * a * b);
    const int matrix =  (global_idx / (a * b)) % depth;
    const int temp = global_idx % (a * b); 
    const int outRow = temp / b;
    const int outCol = temp % b;
    const int xrow = (outRow - diff_row);
    const int xcol = (outCol - diff_col);
    if(xrow >= 0 && xrow < c && xcol >= 0 && xcol < d){
    const int xOffset = batch * depth * c * d + matrix * c * d +  xrow * d + xcol;
    Y[global_idx] += X[xOffset];}
}

__global__ void HeadifyColChannel(const float* __restrict__ input, float* __restrict__ output, const int batch, const int req_channels, const int row, const int col, int flag)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= batch * req_channels * row * col) return;
    const int batch_idx = global_idx / (req_channels*row *col);
    const int channel_idx = (global_idx / (row * col)) % req_channels;
    const int outRow = (global_idx % (row * col)) / col;
    const int outCol = (global_idx % (row * col)) % col;
    if(flag == 0)
    {
        const int input_idx = batch_idx * row * req_channels * col + outRow * req_channels * col + channel_idx  * col + outCol;
        output[global_idx] = input[input_idx];
        return;
    }
    if(flag  == 1)
    {
        const int input_idx = batch_idx * row * req_channels * col + outRow * req_channels * col + channel_idx  * col + outCol;
        output[input_idx] += input[global_idx];
        return;
    }
    if (flag == 2)
    {
        const int full_col = col * req_channels; 
        const int batch_idx   = global_idx / (row * full_col);
        const int outRow      = (global_idx / full_col) % row;
        const int c_out       = global_idx % full_col;
        const int col_idx     = c_out / req_channels;
        const int channel_idx = c_out % req_channels;
        const int input_idx = batch_idx * req_channels * row * col + channel_idx * row * col + outRow * col + col_idx;
        output[global_idx] = input[input_idx];
        return;
    }
    
    if(flag == 3)
    {
        const int full_col = col * req_channels; 
        const int batch_idx   = global_idx / (row * full_col);
        const int outRow      = (global_idx / full_col) % row;
        const int c_out       = global_idx % full_col;
        const int col_idx     = c_out / req_channels;
        const int channel_idx = c_out % req_channels;
        const int input_idx = batch_idx * req_channels * row * col + channel_idx * row * col + outRow * col + col_idx;
        output[input_idx] += input[global_idx];
        return;
    }
}

__global__ void Channel_Concat(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int batch, const int c1, const int c2, const int row, const int col)
{
    const long long global_idx =  threadIdx.x + blockDim.x * blockIdx.x;
    const int outsize = row*col;
    if(global_idx >= batch * (c1+c2)* outsize) return;
    const int batch_idx = global_idx / ((c1+c2)*outsize);
    const int channel_idx = (global_idx / outsize) % (c1+c2);
    const int outRow = (global_idx % outsize) / col;
    const int outCol = (global_idx % outsize) % col;
    if(channel_idx < c1) C[global_idx] = A[batch_idx*c1*outsize+channel_idx*outsize+outRow*col+outCol];
    else C[global_idx] = B[batch_idx*c2*outsize+(channel_idx-c1)*outsize+ outRow*col + outCol];
}

__global__ void Channel_Split(float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C, const int batch, const int c1, const int c2, const int row, const int col)
{
    const long long global_idx =  threadIdx.x + blockDim.x * blockIdx.x;
    const int outsize = row*col;
    if(global_idx >= batch * (c1+c2)* outsize) return;
    const int batch_idx = global_idx / ((c1+c2)*outsize);
    const int channel_idx = (global_idx / outsize) % (c1+c2);
    const int outRow = (global_idx % outsize) / col;
    const int outCol = (global_idx % outsize) % col;
    if(channel_idx < c1) A[batch_idx*c1*outsize+channel_idx*outsize+outRow*col+outCol] += C[global_idx];
    else  B[batch_idx*c2*outsize+(channel_idx-c1)*outsize+outRow*col+outCol] += C[global_idx];
}

__global__ void Column_Concat(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int batch, const int channel, const int row, const int c1, const int c2)
{   
    const long long global_idx =  threadIdx.x + blockDim.x * blockIdx.x;
    const long long A_size = batch * channel * row * c1;
    if(global_idx >= batch * channel * row * (c1+c2)) return;
    if(global_idx < A_size) C[global_idx] = A[global_idx];
    else C[global_idx] = B[global_idx - A_size];
}

__global__ void Column_Split(float* __restrict__ A, float* __restrict__ B, const float* __restrict__ C, const int batch, const int channel, const int row, const int c1, const int c2)
{   
    const long long global_idx =  threadIdx.x + blockDim.x * blockIdx.x;
    const long long A_size = batch * channel * row * c1;
    if(global_idx >= batch * channel * row * (c1+c2)) return;
    if(global_idx < A_size) A[global_idx] += C[global_idx];
    else B[global_idx - A_size] += C[global_idx];
}

__global__ void Vector_Concat(const float* __restrict__ A, float* __restrict__ B, const int batch, const int channel,
                             const int row, const int a_col, const int total_col, const int col_offset) 
{
    const long long idx = threadIdx.x + (long long)blockDim.x * blockIdx.x;
    const long long A_size = (long long)batch * channel * row * a_col;
    if (idx >= A_size) return;

    const int col_a = idx % a_col;
    const int rem1  = idx / a_col;
    const int r     = rem1 % row;
    const int rem2  = rem1 / row;
    const int ch    = rem2 % channel;
    const int b     = rem2 / channel;

    const long long B_idx = (long long)b  * (channel * row * total_col) + (long long)ch * (row * total_col) +
                            (long long)r  * total_col +(col_a + col_offset);
    B[B_idx] = A[idx];
}

__global__ void Vector_Split(const float* __restrict__ dB, float* __restrict__ dA, const int batch, const int channel,
                       const int row, const int a_col, const int total_col, const int col_offset)
{
    const long long idx    = threadIdx.x + (long long)blockDim.x * blockIdx.x;
    const long long dA_size = (long long)batch * channel * row * a_col;
    if (idx >= dA_size) return;

    const int col_a = idx % a_col, rem1  = idx / a_col, r = rem1 % row;
    const int rem2  = rem1 / row, ch = rem2 % channel, b = rem2 / channel;
    const long long dB_idx =(long long)b  * (channel * row * total_col) + (long long)ch * (row * total_col)+
    (long long)r  * total_col +(col_a + col_offset);

    dA[idx] += dB[dB_idx];
}

__global__ void SumSquared(double* __restrict__ scale, const float* __restrict__ grad, const long long total_size)
{
    if (threadIdx.x + blockDim.x * blockIdx.x >= total_size) return;

    const int tpb = blockDim.x;
    const long long block_start = (long long)blockIdx.x * tpb;
    const long long block_end   = min(block_start + tpb, total_size);

    double block_sum = 0.0;
    if (threadIdx.x == 0) 
    {
        for (long long i = block_start; i < block_end; ++i) 
        {
            float g = grad[i];
            if (g != g) {printf("COMPUTATIONAL ERROR: Input to SumSquared is bogus at idx %lld\n", i);}
            double dg = (double)g;
            block_sum += dg * dg;
        }
        atomicAddDouble(scale, block_sum);
    }
}

__global__ void AdamUpdate(float* __restrict__ output, const float* __restrict__ grad, const int total_size, const int t, float* __restrict__ m, float* __restrict__ v,
                           const float b1, const float b2, const float epsilon, const float lr){

    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    const float g = grad[global_idx];
    m[global_idx] = b1 * m[global_idx] + (1.0 - b1) * g;
    v[global_idx] = b2 * v[global_idx] + (1.0 - b2) * g * g;
    const float m_hat = m[global_idx] / (1.0 - powf(b1, t));
    const float v_hat = v[global_idx] / (1.0 - powf(b2, t));
    const float v_hat_scale = sqrtf(v_hat);
    output[global_idx] -= lr * m_hat / (v_hat_scale + epsilon);
}

__global__ void AdamWUpdate(float* __restrict__ output, const float* __restrict__ grad, const int total_size, const int t, float* __restrict__ m, float* __restrict__ v,
                           const float b1, const float b2, const float epsilon, const float lambda, const float lr){

    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    const float g = grad[global_idx];
    m[global_idx] = b1 * m[global_idx] + (1.0 - b1) * g;
    v[global_idx] = b2 * v[global_idx] + (1.0 - b2) * g * g;
    const float m_hat = m[global_idx] / (1.0 - pow(b1, t));
    const float v_hat = v[global_idx] / (1.0 - pow(b2, t));
    const float v_hat_scale = sqrtf(v_hat);
    output[global_idx] -= lr * ((m_hat / (v_hat_scale + epsilon)) + lambda * output[global_idx]);
}

__global__ void KeyUpdate(float* __restrict__ EmbedSpace, const float* __restrict__ grad, const int* __restrict__ keys, const int clen, const int max_clen, 
                          const int embed_dim, const float lr, const long long total)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total) return;
    
    const int batch_idx = global_idx / (clen * embed_dim);
    const int key_row = global_idx % (clen * embed_dim);
    const int token_idx = key_row / embed_dim;
    const int dim_idx = key_row  % embed_dim;
    const int key = keys[batch_idx * max_clen + token_idx];
    if(key == INT_MIN || key < 0) return;
    EmbedSpace[key * embed_dim + dim_idx] += -lr * grad[global_idx];
}

__global__ void OneHotEmbeddings(float* __restrict__ output,const int* __restrict__ keys,int clen,int max_clen,int vocab_size,long long total_size)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int key_row = global_idx / clen;
    const int key_col = global_idx % clen;
    if(global_idx >= total_size) return;
    const int key = keys[key_row * max_clen + key_col];
    if(key == INT_MIN || key < 0 || key >= vocab_size) return;
    //printf("OneHotEmbeddings: Setting output[%lld] to 1 for key %d\n", global_idx * vocab_size + key, key);
    output[global_idx * vocab_size + key] = 1.0f;
}

__global__ void BCompress(const float* __restrict__ X, float* __restrict__ Y, int batch, int channels, int rows, int columns, const float scale)
{

    int col = blockIdx.x; 
    if (col >= columns) return;

    int tid = threadIdx.x;
    int total_rows = batch * channels * rows;
    float sum = 0.0f;
    for (int i = tid; i < total_rows; i += blockDim.x){sum += X[i * columns + col];}
    sum = warpReduceSum(sum);
    static __shared__ float shared_sums[BLOCK_SIZE];
    int lane = tid % BLOCK_SIZE;
    int wid = tid / BLOCK_SIZE;

    if (lane == 0) shared_sums[wid] = sum;
    __syncthreads();
    if (wid == 0)
    {
        sum = (tid < (blockDim.x / BLOCK_SIZE)) ? shared_sums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0){atomicAdd(&Y[col], sum * scale);}
    }
}

__global__ void BCumAdd(float* __restrict__ X, const float* __restrict__ Y, int batch, int channels, int rows, int columns)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= batch * channels * rows) return;
    const int base_offset = idx * columns;
    for(int i = 0; i < columns; ++i){X[base_offset + i] += Y[i];}
}

__global__ void broadcast_add_general(const float* __restrict__ X, const float* __restrict__ bias, float* __restrict__ output, 
                                      const int batch_size, const int channels, const int a, const int b, const int c, const int d)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x; 
    const int output_size = a * b;
    if(global_idx >= batch_size * channels * output_size) return;
    
    const int batch_idx = global_idx / (channels * output_size);
    const int channel_idx = (global_idx / output_size) % channels;
    const int spatial_idx = global_idx % output_size;
    const int row = spatial_idx / b;
    const int col = spatial_idx % b;
    
    // Map to B's spatial dimensions (with broadcasting)
    const int b_row = (c == 1) ? 0 : (row * c / a);  // If c=1, always use row 0
    const int b_col = (d == 1) ? 0 : (col * d / b);  // If d=1, always use col 0
    const int b_idx = batch_idx * channels * c * d + channel_idx * c * d + b_row * d + b_col;
    
    output[global_idx] = X[global_idx] + bias[b_idx];
}

__global__ void broadcast_add_backward(const float* __restrict__ grad_out, float* __restrict__ grad_B, const int batch_size, 
                                       const int channels, const int a, const int b, const int c, const int d)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int B_size = c * d;
    if(global_idx >= batch_size * channels * B_size) return;
    
    const int batch_idx = global_idx / (channels * B_size);
    const int channel_idx = (global_idx / B_size) % channels;
    const int b_spatial_idx = global_idx % B_size;
    const int b_row = b_spatial_idx / d;
    const int b_col = b_spatial_idx % d;
    const int stride_a = a / c; 
    const int stride_b = b / d; 
    
    float sum = 0.0f;
    for(int i = 0; i < stride_a; ++i) {
        for(int j = 0; j < stride_b; ++j) {
            const int a_row = b_row * stride_a + i;
            const int a_col = b_col * stride_b + j;
            const int grad_idx = batch_idx * channels * a * b + 
                                channel_idx * a * b + 
                                a_row * b + a_col;
            sum += grad_out[grad_idx];
        }
    }
    
    grad_B[global_idx] += sum;
}

__global__ void transpose(const float* __restrict__ X, float* __restrict__ output, const int batch_size, const int channels, 
                          const int row, const int col, const int indicator)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x; 
    const long long input_size = row * col;
    const int total_size = batch_size * channels * input_size;
    if (global_idx >= total_size) return;
    const int bc_idx = global_idx / input_size;
    const int inner = global_idx % input_size;
    const int i = inner / col; 
    const int j = inner % col;   
    const int transposed = j * row + i;
    if (indicator == 0) output[bc_idx * input_size + transposed] = X[global_idx];
    else output[bc_idx * input_size + transposed] += X[global_idx];
}

__global__ void broadcast_add(const float* __restrict__ X, const float* __restrict__ bias, float* __restrict__ output, const int batch_size, const int kernels, const int row, const int col)
{
    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x; 
    const int output_size = row * col;
    if(global_idx >= batch_size * kernels * output_size){return;}
    const int matrix =  (global_idx / output_size) % kernels;
    output[global_idx] = X[global_idx] + bias[matrix];
}

__global__ void Accumulate(const float* __restrict__ X, float* __restrict__ Y, const long long total_size, const double scale)
{
    const int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (row_idx >= total_size){return;}
    Y[row_idx] += scale*X[row_idx];

}

__global__ void ScaleAdd(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ Z, const float scale, const long long total_size)
{
    const int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (row_idx >= total_size){return;}
    Z[row_idx] = X[row_idx] + scale * Y[row_idx];
    
}

__global__ void Update(float* __restrict__ output, const float* __restrict__ grad, const int total_size, const double* __restrict__ scale, const float lr)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_idx >= total_size) return;
    const float g = lr * grad[global_idx] / (float)(*scale);
    output[global_idx] -= g;
}

__global__ void Channel_Squeeze1D(const float* __restrict__ X, float* __restrict__ average,
                                   const int batch, const int depth, const int a, const int b)
{
    __shared__ float smem[THREADSPERBLOCK];

    const int channel_idx = blockIdx.x; 
    const int spatial     = a * b;
    const int group_size  = batch * spatial;
    const int tid         = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x)
    {
        const int batch_idx = i / spatial;
        const int s         = i % spatial;
        sum += X[batch_idx * depth * spatial + channel_idx * spatial + s];
    }

    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) average[channel_idx] += smem[0];
}

__global__ void Sqrt_Scale(double* __restrict__ X, const double scale, const int type)
{
    int global_idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(global_idx == 0 && X[0] == X[0])
    {
        const double curr = sqrt(X[0]);
        if (type == 0) X[0] = (curr > 1.0) ? curr * scale : scale;
        else X[0] = (curr > 1.0) ? curr / scale : (double)1.0f / scale;
    }
    else{printf("COMPUTATIONAL ERROR: Input to SQRTSCALE is nan");return;}
}


__global__ void mse_kernel(const float* __restrict__ X, const float* __restrict__ target, float* __restrict__ output, const long long total) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const float tar =   target[global_idx];
    const float x_val = X[global_idx];
    output[global_idx] = (tar - x_val) * (tar - x_val);

}

__global__ void ce_kernel(const float* __restrict__ X, const float* __restrict__ target, float* __restrict__ output, const long long total) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const float tar = target[global_idx];
    const float x_val = X[global_idx];
    output[global_idx] = - tar * logf(x_val + 1e-27f);

}

__global__ void e_kernel(const float* __restrict__ X, float* __restrict__ output, const long long total) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const float x_val = X[global_idx];
    output[global_idx] = -x_val * logf(x_val + 1e-27f);

}


__global__ void scalar_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, float*__restrict__ output, const long long total)
{
    __shared__ double smem[THREADSPERBLOCK];
    const long long gid = threadIdx.x + (long long)blockIdx.x * blockDim.x;
    double val = 0.0;
    if (gid < total) { double d = (double)target[gid] - (double)X[gid]; val = d * d; }
    val = block_reduce_sum(val, smem);
    if (threadIdx.x == 0) atomicAdd(output, (float)(val / (double)total));
}

__global__ void scalar_ce_kernel(const  float* __restrict__ X, const float* __restrict__ target, float*__restrict__ output, const long long total)
{
    __shared__ double smem[THREADSPERBLOCK];
    const long long gid = threadIdx.x + (long long)blockIdx.x * blockDim.x;
    double val = 0.0;
    if (gid < total) val = -(double)target[gid] * log((double)X[gid] + 1e-27);
    val = block_reduce_sum(val, smem);
    if (threadIdx.x == 0) atomicAdd(output, (float)(val / (double)total));
}

__global__ void scalar_e_kernel( const  float* __restrict__ X, float* __restrict__ output, const int batch, const long long total)
{
    __shared__ double smem[THREADSPERBLOCK];
    const long long gid = threadIdx.x + (long long)blockIdx.x * blockDim.x;
    double val = 0.0;
    if (gid < total) { double x = (double)X[gid]; val = -x * log(x + 1e-27f); }
    val = block_reduce_sum(val, smem);
    if (threadIdx.x == 0) atomicAdd(output, (float)(val / (double)batch));
}


__global__ void deriv_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total,const bool last, const bool full) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const float tar = target[global_idx];
    const float x_val = X[global_idx];
    const long long idx = full ? global_idx : 0;
    if(last) output[global_idx] += 2.0f * (x_val - tar) / (float)total;
    else {output[global_idx] += (2.0f * (x_val - tar) * grad[idx]) / (float)total;} // Grad is a single scalar

}

__global__ void deriv_ce_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ grad, float* __restrict__ output, const long long total, const bool last, const bool full) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total){return;}
    const float tar = target[global_idx];
    const float x_val = X[global_idx];
    const long long idx = full ? global_idx : 0;
    if(last) output[global_idx] -=  (tar / (x_val + 1e-27f)) / (float)total;
    else output[global_idx] -=  (tar*grad[idx] / (x_val + 1e-27f)) / (float)total;

}

__global__ void deriv_e_kernel(const float* __restrict__ X, const float* __restrict__ grad, float* __restrict__ output, const int batch, const long long total, const bool last, const bool full) 
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total) return;
    const float x_val = X[global_idx];
    const long long idx = full ? global_idx : 0;
    if(last) output[global_idx] -= (logf(x_val + 1e-27f) + 1.0f) / (float)batch;
    else output[global_idx] -= (logf(x_val + 1e-27f) + 1.0f)*grad[idx] / (float)batch;
}

__global__ void idx_mse_kernel(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ t_idx, float* __restrict__ out, const int batch)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= batch) return;
    const float diff = X[idx * batch + (int)t_idx[idx]] - target[idx];
    atomicAdd(&out[0], diff * diff);
}

__global__ void idx_mse_backward(const float* __restrict__ X, const float* __restrict__ target, const float* __restrict__ t_idx, const float* __restrict__ grad, float* __restrict__ out, const int batch)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= batch) return;
    const float diff = X[idx * batch + (int)t_idx[idx]] - target[idx];
    out[idx] =  diff * diff;
}



__global__ void BatchMinMaxNorm(float* __restrict__ X, const float * __restrict__ max, const float * __restrict__ min, const int batch, const long long total_size) 
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total_size){return;} // 2(x-a)/(b-a) - 1
    const int batch_idx = (global_idx / (total_size / batch));
    const float range = max[batch_idx] - min[batch_idx];
    X[global_idx] = (range > 1e-8) ? 2 * ((X[global_idx] -  min[batch_idx])/(max[batch_idx] - min[batch_idx])) - 1 : 1.0;

}

__global__ void BatchMinMaxDeNorm(float* __restrict__ X,const float max, const float min,const int batch, const long long total_size) // 
{
    const int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total_size){return;} // (y+1)(b-a)/2 + a
    const float range = max-min;
    X[global_idx] = (range > 1e-8f) ? (((X[global_idx]+1.0)*(max-min))/2) + min : min;

}

__global__ void StdNorm(float* __restrict__ X, const float img_max, const float mean, 
                                            const float std, const long long total)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total) return;
    X[idx] = ((X[idx] / img_max) - (float)mean) / (float)std; 
}

__global__ void StdDeNorm(float* __restrict__ X, const float img_max, const float mean, 
                                            const float std, const long long total)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total) return;
    float val = ((X[idx] * std) + mean) * img_max; 
    X[idx] = fmaxf(0.0f, fminf(255.0f, val));
}

__global__ void AddNoise(float* __restrict__ X, float* __restrict__ noise, const int t, const int T, const long long total_size)
{   

    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total_size){return;}
    const double s = 0.008;
    const double t_scale = ((double)t / T) + s;
    const double cost_t = cos((t_scale / (1.0+s))*PIBY2);
    const double cost_t_b = cos((s / (1.0 + s))*PIBY2);
    const double alpha = (cost_t* cost_t) / (cost_t_b*cost_t_b);  
    const double root_beta = sqrt(1.0 - alpha);  
    X[global_idx]  = sqrt(alpha)*X[global_idx] + root_beta * noise[global_idx];
}

__global__ void UnifNoise(float* __restrict__ Y, const long long total_size, const uint64_t seed)
{
    long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size){return;}
    curandStatePhilox4_32_10_t  state;

    curand_init(seed, global_idx, 0, &state);
    const float epsilon = curand_uniform(&state);
    Y[global_idx] = epsilon;
}

__global__ void GaussianNoise(float* __restrict__ Y, const long long total_size, const uint64_t seed)
{
    long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total_size) return;
    curandStatePhilox4_32_10_t  state;

    curand_init(seed, global_idx, 0, &state);
    Y[global_idx] = curand_normal(&state);
}

__global__ void ReplaceNoise(float* __restrict__ X, const float* __restrict__ Y, const float beta, const long long total_size, const uint64_t seed)
{   
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total_size) return;
    curandStatePhilox4_32_10_t  state;
    curand_init(seed, global_idx, 0, &state);
    const float noise = curand_normal(&state);
    X[global_idx] = Y[global_idx] + beta * noise;
}

__global__ void BMax(const float* __restrict__ data, float* __restrict__ value, int batch, int channel, int out_size)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= batch*channel){return;}
    const int batch_idx = global_idx /channel;
    const int channel_idx = global_idx % channel;
    float val = -FLT_MAX;
    for(int i =0; i < out_size; ++i)
    {
    size_t idx = batch_idx * channel*out_size + channel_idx* out_size + i;
     if(data[idx] > val) { val = data[idx];}
    }

    atomicMaxFloat(&value[batch_idx], val);
}

__global__ void BMin(const float* __restrict__ data, float* __restrict__ value, int batch, int channel, int out_size)
{
    const long long global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int batch_idx = global_idx / channel;
    const int channel_idx = global_idx % channel;
    if(global_idx >= batch*channel){return;}
    float val = FLT_MAX;
    for(int i =0; i< out_size; ++i)
    {
    size_t idx = batch_idx * channel*out_size + channel_idx* out_size + i;
     if(data[idx] < val) { val = data[idx];}
    }

    atomicMinFloat(&value[batch_idx], val);
}

__global__ void FindMax(const float* __restrict__ data, float* __restrict__ maxArr,
                        int batch, int channels, int row, int col, int type)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (type == 0) {                           
        if (gid >= batch * channels * row) return;
        int bc = gid / row, r = gid % row;
        float m = -FLT_MAX;
        for (int c = 0; c < col; ++c)
            m = fmaxf(m, data[bc * row * col + r * col + c]);
        maxArr[gid] = m;
    } else {                                   
        if (gid >= batch * channels * col) return;
        int bc = gid / col, c = gid % col;
        float m = -FLT_MAX;
        for (int r = 0; r < row; ++r)
            m = fmaxf(m, data[bc * row * col + r * col + c]);
        maxArr[gid] = m;
    }
}

__global__ void SumRows(const float* __restrict__ data, float* __restrict__ arr,
                        const int batch, const int channels, const int row, const int col)
{
    /*
    Note a very important fact for any future errors, in order to prevent future errors, 
    SumCols and rows ignores all 0s because it was designed for softmax/mask, so if the entire row or col adds to 0, ill be one
    */
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch * channels * col) return;

    int bc = gid / col, c = gid % col;
    float sum = 0.0f;
    for (int r = 0; r < row; ++r)
        sum += data[bc * row * col + r * col + c];
    if (sum < 1e-45f) sum = 1.0f;
    arr[gid] = sum;
}

__global__ void SumCols(const float* __restrict__ data, float* __restrict__ arr,
                        const int batch, const int channels, const int row, const int col)
{
    /*
    Note a very important fact for any future errors, in order to prevent future errors, 
    SumCols and rows ignores all 0s because it was designed for softmax, so if the entire row or col adds to 0, ill be one
    */
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch * channels * row) return;

    int bc = gid / row, r = gid % row;
    float sum = 0.0f;
    for (int c = 0; c < col; ++c)
        sum += data[bc * row * col + r * col + c];
    if (sum < 1e-45f) sum = 1.0f;
    arr[gid] = sum;
}

__global__ void Scale_arr(float* __restrict__ data, const float* __restrict__ arr,
                          const int batch, const int channels, const int row, const int col,
                          const int mode, const int transposed)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch * channels * row * col) return;

    int bc     = global_idx / (row * col);
    int row_in = (global_idx % (row * col)) / col;
    int col_in = (global_idx % (row * col)) % col;

    const int arr_idx = transposed ? bc * col + col_in : bc * row + row_in;
    const float value = arr[arr_idx];
    if (mode == 0) data[global_idx] *= value;
    else           data[global_idx] /= value;
}

__global__ void exponentiate(const float* __restrict__ data, const float* __restrict__ surrogate_deriv, float* __restrict__ output, const long long total_size, const int deriv)
{
    const long long global_idx = threadIdx.x + blockDim.x *blockIdx.x;
    if(global_idx >= total_size) return;
    if(deriv == 1) output[global_idx] += data[global_idx] * surrogate_deriv[global_idx];
    else{ output[global_idx] = expf(data[global_idx]);}
}

__global__ void natural_logarithm(const float* __restrict__ data, const float* __restrict__ surrogate_deriv, float* __restrict__ output, const long long total_size, const int deriv)
{
    const long long global_idx = threadIdx.x + blockDim.x *blockIdx.x;
    if(global_idx >= total_size) return;
    if(deriv == 1) output[global_idx] += surrogate_deriv[global_idx] / data[global_idx];
    else{ output[global_idx] = logf(data[global_idx]);}
}

__global__ void exponentiate(const float* __restrict__ data, float* __restrict__ output,
                             const float* __restrict__ maxArr,
                             const int channels, const int row, const int col,
                             const int type, const long long total_size)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_size) return;

    const int bc = idx / (row * col);
    const int r  = (idx % (row * col)) / col;
    const int c  = (idx % (row * col)) % col;

    const int arr_idx = (type == 0) ? bc * row + r : bc * col + c;
    output[idx] = expf(data[idx] - maxArr[arr_idx]);
}

__global__ void exponentiateM(const float* __restrict__ data, float* __restrict__ output,
                              const float* __restrict__ maxArr,
                              const int channels, const int row, const int col,
                              const int type, const long long total_size)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_size) return;

    const int bc = idx / (row * col);
    const int r  = (idx % (row * col)) / col;
    const int c  = (idx % (row * col)) % col;

    if (c > r) { output[idx] = 0.0f; return; }

    const int arr_idx = (type == 0) ? bc * row + r : bc * col + c;
    output[idx] = expf(data[idx] - maxArr[arr_idx]);
}

__global__ void deriv_softmax(const float* __restrict__ output, const float* __restrict__ grad,
                              float* __restrict__ dinput,
                              const int batch, const int channels, const int row, const int col,
                              const int type)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (type == 0) {
        if (global_idx >= batch * channels * row) return;
        int bc = global_idx / row, r = global_idx % row;

        float dot = 0.0f;
        for (int c = 0; c < col; ++c) {
            int idx = bc * row * col + r * col + c;
            dot += grad[idx] * output[idx];
        }
        for (int c = 0; c < col; ++c) {
            int idx = bc * row * col + r * col + c;
            atomicAdd(&dinput[idx], output[idx] * (grad[idx] - dot));
        }
    } else {
        if (global_idx >= batch * channels * col) return;
        int bc = global_idx / col, c = global_idx % col;

        float dot = 0.0f;
        for (int r = 0; r < row; ++r) {
            int idx = bc * row * col + r * col + c;
            dot += grad[idx] * output[idx];
        }
        for (int r = 0; r < row; ++r) {
            int idx = bc * row * col + r * col + c;
            atomicAdd(&dinput[idx], output[idx] * (grad[idx] - dot));
        }
    }
}

void SoftMax(const float* __restrict__ data, float* __restrict__ arr,
             float* __restrict__ output, float* __restrict__ maxArr,
             const int batch, const int channels, const int row, const int col, const int type)
{
    const int tpb        = THREADSPERBLOCK;
    const long long total_size = (long long)batch * channels * row * col;
    const int max_elems  = (type == 0) ? batch * channels * row
                                       : batch * channels * col;

    FindMax<<<(max_elems+tpb-1) / tpb, tpb>>>(
        data, maxArr, batch, channels, row, col, type);

    exponentiate<<<(total_size + tpb - 1) / tpb, tpb>>>(
        data, output, maxArr, channels, row, col, type, total_size);

    if (type == 0) {
        SumCols<<<(batch * channels * row + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col);
        Scale_arr<<<(total_size + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col, 1, 0);
    } else {
        SumRows<<<(batch * channels * col + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col);
        Scale_arr<<<(total_size + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col, 1, 1);
    }
}

void SoftMask(const float* __restrict__ data, float* __restrict__ arr, float* __restrict__ output, float* __restrict__ maxArr,
              const int batch, const int channels, const int row, const int col, const int type)
{
    const int tpb        = THREADSPERBLOCK;
    const long long total_size = (long long)batch * channels * row * col;
    const int max_elems  = (type == 0) ? batch * channels * row
                                       : batch * channels * col;

    FindMax<<<(max_elems + tpb - 1) / tpb, tpb>>>(
        data, maxArr, batch, channels, row, col, type);

    exponentiateM<<<(total_size + tpb - 1) / tpb, tpb>>>(
        data, output, maxArr, channels, row, col, type, total_size);

    if (type == 0) {
        SumCols<<<(batch * channels * row + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col);
        Scale_arr<<<(total_size + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col, 1, 0);
    } else {
        SumRows<<<(batch * channels * col + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col);
        Scale_arr<<<(total_size + tpb - 1) / tpb, tpb>>>(
            output, arr, batch, channels, row, col, 1, 1);
    }
}

void deriv_SoftMax(const float* __restrict__ output, const float* __restrict__ grad, float* __restrict__ dinput, const int batch, const int channels, const int row, const int col, const int type)
{
    const int tpb = THREADSPERBLOCK;
    const int elems = (type == 0) ? batch * channels * row
                                  : batch * channels * col;
    deriv_softmax<<<(elems + tpb - 1) / tpb, tpb>>>(
        output, grad, dinput, batch, channels, row, col, type);
}

__global__ void ISNAN(const float* __restrict__ X, const long long total)
{
    const long long idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx >= total) return;
    if(X[idx] != X[idx])
    {
        if (idx == 0) printf("NaN at idx=%lld, value=%f\n", idx, X[idx]);
        __trap();
    }
    return;
}

__global__ void GatherEmbeddings(float* __restrict__ output,const float* __restrict__ EmbedSpace, const int* __restrict__ keys, int c,int max_c, int embed_dim,long long total)
{

    const long long global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx >= total) return;
    
    const int batch_idx = global_idx / (c * embed_dim);
    const int key_row = global_idx % (c * embed_dim);
    const int token_idx = key_row / embed_dim;
    const int dim_idx = key_row  % embed_dim;
    const int key = keys[batch_idx * max_c + token_idx];
    
    // For masked positions, output zero
    if(key == INT_MIN || key < 0) {
        output[global_idx] = 0.0f;
    } else {
        output[global_idx] = EmbedSpace[key * embed_dim + dim_idx];
    }
}

__global__ void TopKSampleKernel(const float* __restrict__ arr, int* __restrict__ out_idx, const int size, const int k, const float rand_val)
{
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    float top_vals[THREADSPERBLOCK];
    int   top_idxs[THREADSPERBLOCK];
    int   found = 0;

    for (int i = 0; i < size && found < k; ++i)
    {
        top_vals[found] = arr[i];
        top_idxs[found] = i;
        found++;
    }

    for (int i = 1; i < found; ++i)
    {
        float v = top_vals[i]; int id = top_idxs[i];
        int j = i - 1;
        while (j >= 0 && top_vals[j] < v)
        { top_vals[j+1] = top_vals[j]; top_idxs[j+1] = top_idxs[j]; j--; }
        top_vals[j+1] = v; top_idxs[j+1] = id;
    }
    
    for (int i = k; i < size; ++i)
    {
        if (arr[i] > top_vals[k-1])
        {
            top_vals[k-1] = arr[i];
            top_idxs[k-1] = i;
            int j = k - 1;
            while (j > 0 && top_vals[j] > top_vals[j-1])
            {
                float tv = top_vals[j];   top_vals[j]   = top_vals[j-1]; top_vals[j-1] = tv;
                int   ti = top_idxs[j];   top_idxs[j]   = top_idxs[j-1]; top_idxs[j-1] = ti;
                j--;
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += top_vals[i];
    if (sum < 1e-8f) sum = 1.0f;
    for (int i = 0; i < k; ++i) top_vals[i] /= sum;

    float cumulative = rand_val;
    int   chosen     = top_idxs[k - 1];  
    for (int i = 0; i < k; ++i)
    {
        cumulative -= top_vals[i];
        if (cumulative <= 0.0f) { chosen = top_idxs[i]; break; }
    }

    out_idx[0] = chosen;
}

__global__ void ArgMax(const float* __restrict__ arr, int* __restrict__ X, const int size)
{
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_idx > 0) return;
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    X[0] = max_idx;
}

__global__ void identityKernel(float* __restrict__ input, const int batch, const int channels, const int row_col)
{
    const long long total = (long long)batch * channels * row_col;
    const long long idx = threadIdx.x + blockDim.x* blockIdx.x;
    if(idx >= total) return;
    const int mat = row_col * row_col;
    const int depth = channels * mat;
    const int bc_idx = idx / row_col;
    const int b_idx = bc_idx / channels;
    const int c_idx = bc_idx % channels;
    const int r_idx = idx % row_col;
    input[b_idx * depth + c_idx * mat + r_idx * row_col + r_idx] = 1.0f;
}

__global__ void clampKernel(const float* __restrict__ input, const float* __restrict__ grad, float* __restrict__ output, const float min, const float max, const long long total, const int deriv)
{
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx >= total) return;
    float val = input[global_idx];
    if(deriv == 1) output[global_idx] += (val > min && val < max) ? grad[global_idx] : 0.0f;
    else output[global_idx] = fminf(fmaxf(input[global_idx], min), max);
}

__global__ void minKernel(const float* __restrict__ inp_A, const float* __restrict__ inp_B, float* __restrict__ mask, 
                          float* __restrict__ grad_A, float* __restrict__ grad_B, float* __restrict__ output, const long long size, const int deriv)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size) return;
    const float A = inp_A[idx], B = inp_B[idx]; // 1.0f A, -1.0f B;
    const float val = (A > B) ? B : A;
    if(deriv == 1)
    {
        const float indicator = mask[idx]; // output is a place holder for node->grad
        if(indicator == 1.0f) grad_A[idx] += output[idx];
        if(indicator == -1.0f) grad_B[idx] += output[idx];
    }
    else{mask[idx] = (val == A) ? 1.0f : -1.0f; output[idx] = val;}

}

__global__ void maxKernel(const float* __restrict__ inp_A, const float* __restrict__ inp_B, float* __restrict__ mask, 
                          float* __restrict__ grad_A, float* __restrict__ grad_B, float* __restrict__ output, const long long size, const int deriv)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size) return;
    const float A = inp_A[idx], B = inp_B[idx]; // 1.0f A, -1.0f B;
    const float val = (A < B) ? B : A;
    if(deriv == 1)
    {
        const float indicator = mask[idx]; 
        if(indicator == 1.0f) grad_A[idx] += output[idx];
        if(indicator == -1.0f) grad_B[idx] += output[idx];
    }
    else{mask[idx] = (val == A) ? 1.0f : -1.0f; output[idx] = val;}

}

__global__ void getactionKernel(const float* __restrict__ input, const float* __restrict__ actions, float* __restrict__ output, const int action_dim, const long long total, const bool deriv)
{
    const long long idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= total) return;
    const int ac_id = actions[idx];
    if(deriv) output[idx * action_dim + ac_id] += input[idx];
    else output[idx] = input[idx * action_dim + ac_id];
}

__global__ void static_assign_values( const float* __restrict__ traj, const float* __restrict__ advantages, const float* __restrict__ returns, 
    const float* __restrict__ states, const int*   __restrict__ idx,   float* __restrict__ out_actions, float* __restrict__ out_logp, float* __restrict__ out_adv,
    float* __restrict__ out_ret, float* __restrict__ out_states, const int total, const int state_dim, const int mini_batch_offset)         
{
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= total * state_dim) return;

    const int d = global_idx % state_dim;
    const int i = global_idx / state_dim;
    const int t = idx[mini_batch_offset + i];

    if (d == 0)
    {
        out_actions[i] = traj[t * 5 + 0];
        out_logp[i]    = traj[t * 5 + 2];
        out_adv[i]     = advantages[t];
        out_ret[i]     = returns[t];
    }

    out_states[i * state_dim + d] = states[t * state_dim + d];
}

__global__ void dqn_assign_values(const float* __restrict__ state_replay, const float* __restrict__ replay_traj, float* __restrict__ state, 
    float* __restrict__ target, float* __restrict__ target_indices, const int* __restrict__ d_idx, const int state_total, const int batch, const int start)
    {
        const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (global_idx >= batch * state_total) return;
        const int d = global_idx % state_total;
        const int i = global_idx / state_total;
        const int t = d_idx[start + i];

        if (d == 0)
        {
            target_indices[i] = replay_traj[t * 5 + 0];
            target[i]         = replay_traj[t * 5 + 3];
        }
        
        state[i * state_total + d] = state_replay[t * state_total + d];
    };