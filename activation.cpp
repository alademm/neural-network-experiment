#include "activation.h"
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sigmoid_comp_kernel(float *A, const float *X, int N_rows, int N_cols)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < N_cols && y < N_rows)
    {
        int idx = y*gridDim.x*blockDim.x + x;
        A[idx] = 1.0f / (1.0f + __expf(-X[idx]));
    }
}

Tensor SigmoidActivation::Compute(const Tensor& X)
{
    Tensor result(X.GetNumItems(), X.GetNumChannels(), X.GetNumRows(), X.GetNumCols());

#ifdef USE_GPU
    result.ToGPU();
    dim3 block_size(Tensor::BLOCK_SIZE, Tensor::BLOCK_SIZE);
    dim3 grid_size(X.GetNumPaddedCols() / block_size.x, X.GetNumPaddedRows() / block_size.y);
    const int ch_size = X.GetNumPaddedCols() * X.GetNumPaddedRows();
    const int item_size = X.GetNumChannels() * ch_size;
    for (int it = 0; it < X.GetNumItems(); it++)
    {
        for (int ch = 0; ch < X.GetNumChannels(); ch++)
        {
            float* in_ptr = X.GetGPUPointer() + (it*item_size + ch*ch_size);
            float* out_ptr = result.GetGPUPointer() + (it*item_size + ch*ch_size);
            sigmoid_comp_kernel<<<grid_size, block_size>>>(out_ptr, in_ptr, X.GetNumRows(), X.GetNumCols());
            assert(cudaGetLastError() == cudaSuccess);
        }
    }

#else
    T_ITER_BEGIN(result);
    result(it, ch, r, c) = 1.0f / (1.0f + expf(-X(it, ch, r, c)));
    T_ITER_END(result);
#endif
    return std::move(result);
}

__global__ void sigmoid_deriv_kernel(float *A, const float *X, int N_rows, int N_cols)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < N_cols && y < N_rows)
    {
        int idx = y*gridDim.x*blockDim.x + x;
        float s = 1.0f / (1.0f + __expf(-X[idx]));
        A[idx] = s * (1.0f - s);
    }
}

Tensor SigmoidActivation::Deriv(const Tensor& X)
{
    Tensor result(X.GetNumItems(), X.GetNumChannels(), X.GetNumRows(), X.GetNumCols());

#ifdef USE_GPU
    result.ToGPU();
    dim3 block_size(Tensor::BLOCK_SIZE, Tensor::BLOCK_SIZE);
    dim3 grid_size(X.GetNumPaddedCols() / block_size.x, X.GetNumPaddedRows() / block_size.y);
    const int ch_size = X.GetNumPaddedCols() * X.GetNumPaddedRows();
    const int item_size = X.GetNumChannels() * ch_size;
    for (int it = 0; it < X.GetNumItems(); it++)
    {
        for (int ch = 0; ch < X.GetNumChannels(); ch++)
        {
            float* in_ptr = X.GetGPUPointer() + (it*item_size + ch*ch_size);
            float* out_ptr = result.GetGPUPointer() + (it*item_size + ch*ch_size);
            sigmoid_deriv_kernel<<<grid_size, block_size>>>(out_ptr, in_ptr, X.GetNumRows(), X.GetNumCols());
            assert(cudaGetLastError() == cudaSuccess);
        }
    }
#else
    T_ITER_BEGIN(result);
    result(it, ch, r, c) = 1.0f / (1.0f + expf(-X(it, ch, r, c)));
    result(it, ch, r, c) = result(it, ch, r, c) * (1.0f - result(it, ch, r, c));
    T_ITER_END(result);
#endif 
    return std::move(result);
}
