#pragma once

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include <vector>

class Net
{
public:
    Net(Loss *loss, float lr);
    ~Net();

    inline Net& operator<< (Layer *l)
    {
        m_layers.push_back(l); return *this;
    }

    Tensor Forward(const Tensor &X);

    void SGD(const std::vector<Tensor> &training_data, const std::vector<Tensor> &training_labels,
        int num_epochs, int mini_batch_size,
        std::vector<Tensor> *test_data = nullptr, std::vector<Tensor> *test_labels = nullptr);

private:
    void TrainMinibatch(Tensor &X, const Tensor &y);
    int Evaluate(const std::vector<Tensor> &test_data, const std::vector<Tensor> &test_labels);
    std::vector<Layer*> m_layers;
    Loss *m_loss;
    float m_lr;
};
