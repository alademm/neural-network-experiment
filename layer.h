#pragma once

#include "tensor.h"

class Layer
{
public:
    virtual ~Layer() {}
    virtual void Forward(Tensor &x) = 0;
    virtual void Backward(Tensor &err) = 0;
    virtual void UpdateWeights(float lr) = 0;
};
