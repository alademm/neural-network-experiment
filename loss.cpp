#include "loss.h"

Tensor MSELoss::Compute(const Tensor& y_pred, const Tensor & y)
{
    assert(0 && "NO IMPLEMENTED");
    return Tensor();
}

Tensor MSELoss::Deriv(const Tensor& y_pred, const Tensor & y)
{
    return (y_pred - y);
}