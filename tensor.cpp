#include "Tensor.h"
#include <algorithm>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256

Tensor::Tensor() : m_data(nullptr), m_data_gpu(nullptr), m_nitems(0), m_nchannels(0), m_nrows(0), m_ncols(0), m_npadded_rows(0), m_npadded_cols(0)
{
}

Tensor::Tensor(int n_items, int n_channels, int n_rows, int n_cols) :
    m_nitems(n_items), m_nchannels(n_channels), m_nrows(n_rows), m_ncols(n_cols), m_data(nullptr), m_data_gpu(nullptr)
{
    m_npadded_rows = ((m_nrows + BLOCK_SIZE - 1)/BLOCK_SIZE) * BLOCK_SIZE;
    m_npadded_cols = ((m_ncols + BLOCK_SIZE - 1)/BLOCK_SIZE) * BLOCK_SIZE;
    allocate_storage(GetNumTotalElements());
}

Tensor::~Tensor()
{
    free_storage();
}

Tensor::Tensor(Tensor &&t2) : m_data(nullptr), m_data_gpu(nullptr), m_nitems(0), m_nchannels(0), m_nrows(0), m_ncols(0), m_npadded_rows(0), m_npadded_cols(0)
{
    m_nitems = t2.m_nitems;
    m_nchannels = t2.m_nchannels;
    m_nrows = t2.m_nrows;
    m_ncols = t2.m_ncols;
    m_npadded_rows = t2.m_npadded_rows;
    m_npadded_cols = t2.m_npadded_cols;
    m_data = t2.m_data;
    m_data_gpu = t2.m_data_gpu;

    t2.m_nitems = 0;
    t2.m_nchannels = 0;
    t2.m_nrows = 0;
    t2.m_ncols = 0;
    t2.m_npadded_rows = 0;
    t2.m_npadded_cols = 0;
    t2.m_data = nullptr;
    t2.m_data_gpu = nullptr;
}

Tensor &Tensor::operator=(Tensor &&t2)
{
    if (this != &t2)
    {
        free_storage();
        m_nitems = t2.m_nitems;
        m_nchannels = t2.m_nchannels;
        m_nrows = t2.m_nrows;
        m_ncols = t2.m_ncols;
        m_npadded_rows = t2.m_npadded_rows;
        m_npadded_cols = t2.m_npadded_cols;
        m_data = t2.m_data;
        m_data_gpu = t2.m_data_gpu;

        t2.m_nitems = 0;
        t2.m_nchannels = 0;
        t2.m_nrows = 0;
        t2.m_ncols = 0;
        t2.m_npadded_rows = 0;
        t2.m_npadded_cols = 0;
        t2.m_data = nullptr;
        t2.m_data_gpu = nullptr;
    }
    return *this;
}

void Tensor::allocate_storage(size_t n_elements)
{
    assert(m_data == nullptr);
    m_data = (float *)malloc(sizeof(float) * n_elements);
    for (size_t i = 0; i < n_elements; i++)
    {
        m_data[i] = 0.0f;
    }
}

void Tensor::free_storage()
{
    if (m_data != nullptr)
    {
        free((void *)m_data);
        m_data = nullptr;
    }

    if (m_data_gpu != nullptr)
    {
        cudaFree(m_data_gpu);
        m_data_gpu = nullptr;
    }
}

Tensor Tensor::Clone()const
{
    Tensor c(GetNumItems(), GetNumChannels(), GetNumRows(), GetNumCols());
    std::copy(m_data, m_data + GetNumTotalElements(), c.m_data);
    if (m_data_gpu != nullptr)
    {
        c.ToGPU();
    }
    return std::move(c);
}

bool Tensor::ToGPU()
{
    assert(m_data_gpu == nullptr);
    cudaError_t err = cudaMalloc((void**)&m_data_gpu, GetNumTotalElements() * sizeof(float));
    if (err != cudaSuccess)
    {
        assert(0 && "Allocating memory on GPU failed.");
        cudaFree(m_data_gpu);
        m_data_gpu = nullptr;
        return false;
    }

    err = cudaMemcpy(m_data_gpu, m_data, GetNumTotalElements() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        assert(0 && "Copying data to GPU failed.");
        cudaFree(m_data_gpu);
        m_data_gpu = nullptr;
        return false;
    }

    return true;
}

bool Tensor::UpdateFromGPU()
{
    assert(m_data_gpu != nullptr);
    cudaError_t err = cudaMemcpy(m_data, m_data_gpu, GetNumTotalElements() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        assert(0 && "UpdateFromGPU failed");
        return false;
    }
    return true;
}

bool Tensor::UpdateFromHost()
{
    assert(m_data_gpu != nullptr);
    cudaError_t err = cudaMemcpy(m_data_gpu, m_data, GetNumTotalElements() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        assert(0 && "UpdateFromHost failed");
        return false;
    }
    return true;
}

// Taken from matrixMul.cu sample in NVIDIA GPU SDK: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/matrixMul
template <int BLOCK_SIZE> __global__ void mat_mul_ab(float *C, const float *A, const float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

Tensor Tensor::operator*(const Tensor &t2)const
{
    assert(GetNumCols() == t2.GetNumRows());
    assert(GetNumChannels() == t2.GetNumChannels());
#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    assert(t2.m_data_gpu != nullptr);
    //cudaStreamSynchronize((cudaStream_t)0);
    const int item_size_a = m_nchannels * m_npadded_rows * m_npadded_cols;
    const int item_size_b = t2.m_nchannels * t2.m_npadded_rows * t2.m_npadded_cols;
    const int ch_size_a = m_npadded_rows * m_npadded_cols;
    const int ch_size_b = t2.m_npadded_rows * t2.m_npadded_cols;
    const int item_size_c = m_nchannels * m_npadded_rows * t2.m_npadded_cols;
    const int ch_size_c = m_npadded_rows * t2.m_npadded_cols;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(t2.m_npadded_cols / threads.x, m_npadded_rows / threads.y);

    if (GetNumItems() == t2.GetNumItems())
    {
        Tensor result(GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        result.ToGPU();
        //std::vector<cudaStream_t> streams(GetNumItems());
        for (int it = 0; it < GetNumItems(); it++)
        {
            //cudaStreamCreate(&streams[it]);
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                const float * A = m_data_gpu + (it*item_size_a + ch*ch_size_a);
                const float * B = t2.m_data_gpu + (it*item_size_b + ch*ch_size_b);
                float *C = result.m_data_gpu + (it*item_size_c + ch*ch_size_c);
                mat_mul_ab<BLOCK_SIZE> <<<grid, threads>>>(C, A, B, m_npadded_cols, t2.m_npadded_cols);
            }
        }

        //for (int i = 0; i < streams.size(); i++)
        //{
        //    cudaStreamSynchronize(streams[i]);
        //    cudaStreamDestroy(streams[i]);
        //}

        return result;
    }

    if (GetNumItems() != 1 && t2.GetNumItems() != 1)
    {
        return Tensor();
    }

    if (GetNumItems() == 1)
    {
        Tensor result(t2.GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        result.ToGPU();
        //std::vector<cudaStream_t> streams(t2.GetNumItems());
        for (int it = 0; it < t2.GetNumItems(); it++)
        {
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                const float * A = m_data_gpu + (ch*ch_size_a);
                const float * B = t2.m_data_gpu + (it*item_size_b + ch*ch_size_b);
                float *C = result.m_data_gpu + (it*item_size_c + ch*ch_size_c);
                mat_mul_ab<BLOCK_SIZE> <<<grid, threads>>>(C, A, B, m_npadded_cols, t2.m_npadded_cols);
            }
        }

        //for (int i = 0; i < streams.size(); i++)
        //{
        //    cudaStreamSynchronize(streams[i]);
        //    cudaStreamDestroy(streams[i]);
        //}

        return result;
    }

    if (t2.GetNumItems() == 1)
    {
        Tensor result(GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        result.ToGPU();
        //std::vector<cudaStream_t> streams(GetNumItems());
        for (int it = 0; it < GetNumItems(); it++)
        {
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                const float * A = m_data_gpu + (it*item_size_a + ch*ch_size_a);
                const float * B = t2.m_data_gpu + (ch*ch_size_b);
                float *C = result.m_data_gpu + (it*item_size_c + ch*ch_size_c);
                mat_mul_ab<BLOCK_SIZE> <<<grid, threads>>>(C, A, B, m_npadded_cols, t2.m_npadded_cols);
            }
        }

        //for (int i = 0; i < streams.size(); i++)
        //{
        //    cudaStreamSynchronize(streams[i]);
        //    cudaStreamDestroy(streams[i]);
        //}

        return result;
    }

#else
    if (GetNumItems() == t2.GetNumItems())
    {
        Tensor result(GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        for (int it = 0; it < GetNumItems(); it++)
        {
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                for (int i = 0; i < GetNumRows(); i++)
                {
                    for (int j = 0; j < t2.GetNumCols(); j++)
                    {
                        result(it, ch, i, j) = 0;
                        for (int k = 0; k < GetNumCols(); k++)
                        {
                            result(it, ch, i, j) += ((*this)(it, ch, i, k) * t2(it, ch, k, j));
                        }
                    }
                }
            }
        }
        return result;
    }

    if (GetNumItems() != 1 && t2.GetNumItems() != 1)
    {
        return Tensor();
    }

    if (GetNumItems() == 1)
    {
        Tensor result(t2.GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        for (int it = 0; it < t2.GetNumItems(); it++)
        {
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                for (int i = 0; i < GetNumRows(); i++)
                {
                    for (int j = 0; j < t2.GetNumCols(); j++)
                    {
                        result(it, ch, i, j) = 0;
                        for (int k = 0; k < GetNumCols(); k++)
                        {
                            result(it, ch, i, j) += ((*this)(0, ch, i, k) * t2(it, ch, k, j));
                        }
                    }
                }
            }
        }
        return result;
    }

    if (t2.GetNumItems() == 1)
    {
        Tensor result(GetNumItems(), GetNumChannels(), GetNumRows(), t2.GetNumCols());
        for (int it = 0; it < GetNumItems(); it++)
        {
            for (int ch = 0; ch < GetNumChannels(); ch++)
            {
                for (int i = 0; i < GetNumRows(); i++)
                {
                    for (int j = 0; j < t2.GetNumCols(); j++)
                    {
                        result(it, ch, i, j) = 0;
                        for (int k = 0; k < GetNumCols(); k++)
                        {
                            result(it, ch, i, j) += ((*this)(it, ch, i, k) * t2(0, ch, k, j));
                        }
                    }
                }
            }
        }
        return result;
    }
#endif
}

__global__ void subtraction_ab_kernel(float *C, const float *A, const float *B, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] - B[idx];
    }
}

Tensor Tensor::operator-(const Tensor& t2)const
{
    assert(GetNumItems() == t2.GetNumItems());
    assert(GetNumChannels() == t2.GetNumChannels());
    assert(GetNumRows() == t2.GetNumRows());
    assert(GetNumCols() == t2.GetNumCols());
    assert(GetNumTotalElements() == t2.GetNumTotalElements());

    Tensor result(GetNumItems(), GetNumChannels(), GetNumRows(), GetNumCols());

#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    assert(t2.m_data_gpu != nullptr);
    result.ToGPU();
    int N = GetNumTotalElements();
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    subtraction_ab_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(result.m_data_gpu, this->m_data_gpu, t2.m_data_gpu, N);
    assert(cudaGetLastError() == cudaSuccess);

#else
    for (int i = 0, count = GetNumTotalElements(); i < count; i++)
    {
        result(i) = (*this)(i) - t2(i);
    }
#endif

    return result;
}

__global__ void subtraction_a_kernel(float *A, const float *B, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] -= B[idx];
    }
}

void Tensor::operator-=(const Tensor& t2)
{
    assert(GetNumItems() == t2.GetNumItems());
    assert(GetNumChannels() == t2.GetNumChannels());
    assert(GetNumRows() == t2.GetNumRows());
    assert(GetNumCols() == t2.GetNumCols());

#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    assert(t2.m_data_gpu != nullptr);
    int N = GetNumTotalElements();
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    subtraction_a_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(this->m_data_gpu, t2.m_data_gpu, N);
    assert(cudaGetLastError() == cudaSuccess);

#else
    for (int i = 0, count = GetNumTotalElements(); i < count; i++)
    {
        m_data[i] -= t2(i);
    }
#endif
}

__global__ void mat_scalar_mul_kernel(float *A, float m, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] *= m;
    }
}

void Tensor::operator*=(const float m)
{
#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    int N = GetNumTotalElements();
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mat_scalar_mul_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(this->m_data_gpu, m, N);
    assert(cudaGetLastError() == cudaSuccess);
#else
    for (int i = 0, count = GetNumTotalElements(); i < count; i++)
    {
        m_data[i] *= m;
    }
#endif
}

__global__ void set_zero_kernel(float *A, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] = 0.0f;
    }
}

void Tensor::SetZero()
{
#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    int N = GetNumTotalElements();
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_zero_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(this->m_data_gpu, N);
    assert(cudaGetLastError() == cudaSuccess);
#else
    for (int i = 0, count = GetNumTotalElements(); i < count; i++)
    {
        m_data[i] = 0.0f;
    }
#endif
}

__global__ void hadamard_product_kernel(float *A, const float *B, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] *= B[idx];
    }
}

void Tensor::HadamardProduct(const Tensor &t2)
{
    assert(GetNumItems() == t2.GetNumItems());
    assert(GetNumChannels() == t2.GetNumChannels());
    assert(GetNumRows() == t2.GetNumRows());
    assert(GetNumCols() == t2.GetNumCols());
#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    assert(t2.m_data != nullptr);
    int N = GetNumTotalElements();
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hadamard_product_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(this->m_data_gpu, t2.m_data_gpu, N);
    assert(cudaGetLastError() == cudaSuccess);
#else
    for (int i = 0, count = GetNumTotalElements(); i < count; i++)
    {
        m_data[i] *= t2(i);
    }
#endif
}

template<int BLOCK_SIZE> __global__ void transpose_kernel(float *T, const float *X)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];
    int i_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int i_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int w = gridDim.x * BLOCK_SIZE;
    int h = gridDim.y * BLOCK_SIZE;

    tile[threadIdx.y][threadIdx.x] = X[i_y * w + i_x];
    __syncthreads();

    i_x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    i_y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    T[i_y*h + i_x] = tile[threadIdx.x][threadIdx.y];
}

Tensor Tensor::T()
{
    Tensor t(GetNumItems(), GetNumChannels(), GetNumCols(), GetNumRows());

#ifdef USE_GPU  
    t.ToGPU();
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(m_npadded_cols/BLOCK_SIZE, m_npadded_rows/BLOCK_SIZE);
    const int ch_size = m_npadded_cols * m_npadded_rows;
    const int item_size = m_nchannels * ch_size;
    for (int it = 0; it < GetNumItems(); it++)
    {
        for (int ch = 0; ch < GetNumChannels(); ch++)
        {
            int idx = it*item_size + ch*ch_size;
            transpose_kernel<BLOCK_SIZE> <<<grid_size, block_size>>>((t.m_data_gpu+idx), (this->m_data_gpu+idx));
            assert(cudaGetLastError() == cudaSuccess);
        }
    }
#else
    for (int it = 0; it < GetNumItems(); it++)
    {
        for (int ch = 0; ch < GetNumChannels(); ch++)
        {
            for (int i = 0; i < GetNumCols(); i++)
            {
                for (int j = 0; j < GetNumRows(); j++)
                {
                    t(it, ch, i, j) = (*this)(it, ch, j, i);
                }
            }
        }
    }
#endif
    return t;
}

__global__ void set_item_kernel(float *A, const float * B, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] = B[idx];
    }
}

void Tensor::SetItemHost(int idx, const Tensor& val)
{
    assert(GetNumRows() == val.GetNumRows());
    assert(GetNumCols() == val.GetNumCols());
    assert(GetNumChannels() == val.GetNumChannels());
    assert(val.GetNumItems() == 1);
    const int item_size = m_nchannels * m_npadded_rows * m_npadded_cols;
    const int item_idx = idx * item_size;
    for (int i = item_idx, j = 0; j < item_size; i++, j++)
    {
        m_data[i] = val.m_data[j];
    }
}

Tensor Tensor::GetItemHost(int idx)const
{
    Tensor t(1, GetNumChannels(), GetNumRows(), GetNumCols());
    const int item_size = m_nchannels * m_npadded_rows * m_npadded_cols;
    const int item_idx = idx * item_size;
    for (int i = item_idx, j = 0; i < (item_idx + item_size); i++, j++)
    {
        t.m_data[j] = m_data[i];
    }
    return t;
}

__global__ void sum_kernel(float *A, const float * B, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] += B[idx];
    }
}
Tensor Tensor::ItemsSummed()const
{
    Tensor t(1, GetNumChannels(), GetNumRows(), GetNumCols());
    const int item_size = m_nchannels * m_npadded_rows * m_npadded_cols;

#ifdef USE_GPU
    assert(m_data_gpu != nullptr);
    t.ToGPU();
    const int N = item_size;
    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int it = 0; it < GetNumItems(); it++)
    {
        sum_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(t.m_data_gpu, (this->m_data_gpu + (it * item_size)), N);
        assert(cudaGetLastError() == cudaSuccess);
    }
#else
    for (int it = 0; it < GetNumItems(); it++)
    {
        const int item_idx = it * item_size;
        for (int i = item_idx, j = 0; i < (item_idx + item_size); i++, j++)
        {
            t.m_data[j] += m_data[i];
        }
    }
#endif 
    return std::move(t);
}