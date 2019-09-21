#pragma once

#include <assert.h>
#include <utility> 

//#define USE_GPU

//class Tensor;
//class TElement
//{
//public:
//    operator float()const { return m_t->m_data[m_idx]; }
//    void operator=(float val)
//    {
//        m_t->m_data[m_idx] = val;
//        m_t->m_valid[m_idx] = true;
//    }
//
//private:
//    friend class Tensor;
//    explicit TElement(Tensor *t): m_t(t) {}
//    Tensor *m_t;
//    int m_idx;
//};

class Tensor
{
public:
    enum { BLOCK_SIZE = 32 };

    Tensor();
    Tensor(int n_items, int n_channels, int n_rows, int n_cols);
    ~Tensor();
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&);
    Tensor &operator=(Tensor &&);

    Tensor Clone()const;

    bool ToGPU();
    bool UpdateFromGPU();
    bool UpdateFromHost();

    Tensor operator*(const Tensor &t2)const;
    Tensor operator-(const Tensor& t2)const;
    void operator*=(const float m);
    void operator-=(const Tensor& t2);

    // The following index into the host array, so make sure to have it updated from GPU first.
    inline float &operator()(int item, int channel, int row, int col);
    inline float operator()(int item, int channel, int row, int col)const;
    inline float& operator()(int idx);
    inline float operator()(int idx) const;

    int GetNumItems()const { return m_nitems; }
    int GetNumChannels()const { return m_nchannels; }
    int GetNumCols()const { return m_ncols; }
    int GetNumPaddedCols()const { return m_npadded_cols; }
    int GetNumPaddedRows()const { return m_npadded_rows; }
    int GetNumRows()const { return m_nrows; }
    int GetNumTotalElements()const { return (m_nitems * m_nchannels * m_npadded_rows * m_npadded_cols); }
    int GetNumValidElements()const { return (m_nitems * m_nchannels * m_nrows * m_ncols); }

    void SetItemHost(int idx, const Tensor& val);
    Tensor GetItemHost(int idx)const;

    Tensor ItemsSummed()const;

    void SetZero();
    void HadamardProduct(const Tensor &t2);
    Tensor T();

    inline float* GetGPUPointer()const { return m_data_gpu; }

private:
    void allocate_storage(size_t n_elements);
    void free_storage();
    float *m_data, *m_data_gpu;
    int m_nitems, m_nchannels, m_nrows, m_ncols, m_npadded_rows, m_npadded_cols;
};

#define T_ITER_BEGIN(T) \
    for (int it = 0; it < T.GetNumItems(); it++) {\
        for (int ch = 0; ch < T.GetNumChannels(); ch++) {\
            for (int r = 0; r < T.GetNumRows(); r++) {\
                for (int c = 0; c < T.GetNumCols(); c++) {\

#define T_ITER_END(T) }}}}\

//================================================================
float &Tensor::operator()(int item, int channel, int row, int col)
{
    assert(item >= 0 && item < GetNumItems());
    assert(channel >= 0 && channel < GetNumChannels());
    assert(row >= 0 && row < GetNumRows());
    assert(col >= 0 && col < GetNumCols());
    return m_data[item*m_nchannels*m_npadded_rows*m_npadded_cols + channel*m_npadded_rows*m_npadded_cols + row*m_npadded_cols + col];
}

float Tensor::operator()(int item, int channel, int row, int col)const
{
    assert(item >= 0 && item < GetNumItems());
    assert(channel >= 0 && channel < GetNumChannels());
    assert(row >= 0 && row < GetNumRows());
    assert(col >= 0 && col < GetNumCols());
    return m_data[item*m_nchannels*m_npadded_rows*m_npadded_cols + channel*m_npadded_rows*m_npadded_cols + row*m_npadded_cols + col];
}

float& Tensor::operator()(int idx)
{
    assert(idx >= 0 && idx < GetNumTotalElements());
    return m_data[idx];
}

float Tensor::operator()(int idx) const
{
    assert(idx >= 0 && idx < GetNumTotalElements());
    return m_data[idx];
}
