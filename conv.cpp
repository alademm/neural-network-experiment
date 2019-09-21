#include "conv.h"
#include <random>
#include <chrono>

static void initialize_filter_bank(Tensor &filters)
{
}

Conv::Conv(int in_channels, int out_channels, int filter_size, int stride, Activation *act) :
    m_in_channels(in_channels), m_out_channels(out_channels), m_filter_size(filter_size), m_stride(stride), m_activation(act)
{
    m_filters = Tensor(m_out_channels, m_in_channels, m_filter_size, m_filter_size);
    m_grad = Tensor(m_out_channels, m_in_channels, m_filter_size, m_filter_size);
    initialize_filter_bank(m_filters);
}

Conv::~Conv()
{
    delete m_activation;
}

static float dot_product(const Tensor& filters, int filter_idx, const Tensor& x, int b_it, int i, int j, int filter_size)
{
    float dotp = 0.0f;
    for (int d = 0; d < filters.GetNumChannels(); d++)
    {
        for (int fi = 0; fi < filter_size; fi++, i++)
        {
            for (int fj = 0; fj < filter_size; fj++, j++)
            {
                dotp += (filters(filter_idx, d, fi, fj) * x(b_it, d, i, j));
            }
        }
    }
    return dotp;
}

void Conv::Forward(Tensor &x)
{
    assert((x.GetNumRows() - m_filter_size) % m_stride == 0);
    assert((x.GetNumCols() - m_filter_size) % m_stride == 0);

    m_z = Tensor(x.GetNumItems(), m_out_channels, (x.GetNumRows()-m_filter_size)/m_stride + 1, (x.GetNumCols()-m_filter_size)/m_stride + 1);
    for (int it = 0; it < x.GetNumItems(); it++)
    {
        for (int ch = 0; ch < m_filters.GetNumItems(); ch++)
        {
            for (int i = 0; i <= (x.GetNumRows() - m_filter_size); i += m_stride)
            {
                for (int j = 0; j <= (x.GetNumCols() - m_filter_size); j += m_stride)
                {
                    m_z(it, ch, (i/m_stride), (j/m_stride)) = dot_product(m_filters, ch, x, it, i, j, m_filter_size);
                }
            }
        }
    }
    m_input_a = std::move(x);
    x = m_activation->Compute(m_z);
}

void Conv::Backward(Tensor &err)
{
    m_grad.SetZero(); // to start accumelating gradient
    
}

void Conv::UpdateWeights(float lr)
{
}
