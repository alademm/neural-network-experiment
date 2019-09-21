#include "fully_connected.h"
#include <random>
#include <chrono>

FullyConnected::FullyConnected(int d_in, int d_out, Activation *act) : m_activation(act)
{
    m_W = Tensor(1, 1, d_out, d_in);
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    T_ITER_BEGIN(m_W);
    m_W(it, ch, r, c) = dist(gen);
    T_ITER_END(m_W);

#ifdef USE_GPU
    m_W.ToGPU();
#endif
}

FullyConnected::~FullyConnected()
{
    delete m_activation;
}

void FullyConnected::Forward(Tensor &x)
{
    m_z = m_W * x;
    m_prev_a = std::move(x);
    x = m_activation->Compute(m_z);
}

void FullyConnected::Backward(Tensor &err)
{
    Tensor& err_l = err;
    err_l.HadamardProduct(m_activation->Deriv(m_z));
    m_grad = err_l * m_prev_a.T();
    err = m_W.T() * err_l;
}

void FullyConnected::UpdateWeights(float lr)
{
    m_grad = m_grad.ItemsSummed();
    m_grad *= lr;
    m_W -= m_grad;
}
