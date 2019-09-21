#pragma once

#include "layer.h"
#include "tensor.h"
#include "activation.h"

class Conv : public Layer
{
public:
    Conv(int in_channels, int out_channels, int filter_size, int stride, Activation *act);
    ~Conv();
    virtual void Forward(Tensor &x);
    virtual void Backward(Tensor &err);
    virtual void UpdateWeights(float lr);

private:
    Tensor m_filters, m_input_a, m_z, m_grad;
    int m_in_channels, m_out_channels, m_filter_size, m_stride;
    Activation *m_activation;
};
