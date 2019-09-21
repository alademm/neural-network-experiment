#include "net.h"
#include <random>
#include <numeric>
#include <iostream>

Net::Net(Loss *loss, float lr) : m_loss(loss), m_lr(lr)
{
}

Net::~Net()
{
    for (Layer* l : m_layers)
    {
        delete l;
    }
    m_layers.clear();
    delete m_loss;
}

Tensor Net::Forward(const Tensor &in_X)
{
    Tensor X = in_X.Clone();
#ifdef USE_GPU
    X.ToGPU();
#endif
    for (Layer *l : m_layers)
    {
        l->Forward(X);
    }
    return X;
}

void Net::SGD(const std::vector<Tensor> &training_data, const std::vector<Tensor> &training_labels,
    int num_epochs, int mini_batch_size,
    std::vector<Tensor> *test_data, std::vector<Tensor> *test_labels)
{
    std::vector<int> train_data_indices(training_data.size());
    std::iota(train_data_indices.begin(), train_data_indices.end(), 0);
    auto rng = std::default_random_engine{};

    for (int i = 0; i < num_epochs; i++)
    {
        std::shuffle(train_data_indices.begin(), train_data_indices.end(), rng);
        size_t start_idx = 0;
        size_t end_idx = mini_batch_size;
        while (true)
        {
            Tensor batch_data(mini_batch_size, 1, 28*28, 1);
            Tensor batch_labels(mini_batch_size, 1, 10, 1);
            for (size_t k = start_idx; k < end_idx; k++)
            {
                batch_data.SetItemHost(k%mini_batch_size, training_data[train_data_indices[k]]);
                batch_labels.SetItemHost(k%mini_batch_size, training_labels[train_data_indices[k]]);
            }
#ifdef USE_GPU
            batch_data.ToGPU();
            batch_labels.ToGPU();
#endif
            TrainMinibatch(batch_data, batch_labels);
            start_idx += mini_batch_size;
            if (start_idx >= training_data.size())
            {
                break;
            }
            end_idx += mini_batch_size;
            end_idx = std::min(end_idx, training_data.size());
            int bs = end_idx - start_idx;
            if (bs < mini_batch_size)
            {
                batch_data = Tensor(bs, 1, 28*28, 1);
                batch_labels = Tensor(bs, 1, 10, 1);
#ifdef USE_GPU
                batch_data.ToGPU();
                batch_labels.ToGPU();
#endif
            }
        }

        std::cout << "Epoch #" << i << " finished. ";
        if (test_data)
        {
            std::cout << Evaluate(*test_data, *test_labels) << "/" << test_data->size();
        }
        std::cout << std::endl;
    }
}

void Net::TrainMinibatch(Tensor &in_X, const Tensor &in_y)
{
    Tensor &X = in_X;
    const Tensor& y = in_y;

    for (Layer *l : m_layers)
    {
        l->Forward(X);
    }

    Tensor err = m_loss->Deriv(X, y);
    for (int i = m_layers.size()-1; i >= 0; i--)
    {
        m_layers[i]->Backward(err);
        m_layers[i]->UpdateWeights(m_lr);
    }
}

static int GetMaxIdx(const Tensor &y)
{
    assert(y.GetNumCols() == 1);
    assert(y.GetNumChannels() == 1);
    assert(y.GetNumItems() == 1);
    int idx = -1;
    float max_activation = std::numeric_limits<float>::lowest();
    for (int i = 0, count = y.GetNumRows(); i < count; i++)
    {
        if (y(0, 0, i, 0) > max_activation)
        {
            max_activation = y(0, 0, i, 0);
            idx = i;
        }
    }
    return idx;
}

int Net::Evaluate(const std::vector<Tensor> &test_data, const std::vector<Tensor> &test_labels)
{
    int num_correct = 0;
    for (int i = 0; i < test_data.size(); i++)
    {
        Tensor prediction = Forward(test_data[i]);
#ifdef USE_GPU
        prediction.UpdateFromGPU();
#endif
        int predicted_number = GetMaxIdx(prediction);
        int correct_number = GetMaxIdx(test_labels[i]);
        //std::cout << "Predicted: " << predicted_number << std::endl;
        //std::cout << "Correct: " << correct_number << std::endl;
        if (predicted_number == correct_number)
        {
            num_correct++;
        }
    }
    return num_correct;
}