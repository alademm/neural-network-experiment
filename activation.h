#pragma once

#include "tensor.h"

class Activation
{
public:
    virtual ~Activation() {}
    virtual Tensor Compute(const Tensor& X) = 0;
    virtual Tensor Deriv(const Tensor& X) = 0;
};

class SigmoidActivation : public Activation
{
public:
    virtual Tensor Compute(const Tensor& X);
    virtual Tensor Deriv(const Tensor& X);
};
