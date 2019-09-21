#pragma once

#include "layer.h"
#include "tensor.h"
#include "activation.h"

class FullyConnected : public Layer
{
public:
    FullyConnected(int d_in, int d_out, Activation *act);
    ~FullyConnected();
    virtual void Forward(Tensor &x);
    virtual void Backward(Tensor &err);
    virtual void UpdateWeights(float lr);

private:
    Tensor m_W, m_grad, m_z, m_prev_a;
    Activation *m_activation;
};
