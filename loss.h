#pragma once

#include "tensor.h"

class Loss
{
public:
    virtual ~Loss() {}
    virtual Tensor Compute(const Tensor& y_pred, const Tensor & y) = 0;
    virtual Tensor Deriv(const Tensor& y_pred, const Tensor & y) = 0;
};

class MSELoss : public Loss
{
public:
    virtual Tensor Compute(const Tensor& y_pred, const Tensor & y);
    virtual Tensor Deriv(const Tensor& y_pred, const Tensor & y);
};