#include "neural_network.h"
#include <iostream>
#include <numeric>
#include <chrono>
#include <random>
#include <math.h>
#include <string>
#include <sstream>
//#include <CL/cl.h>

// class OpenCLHandler
// {
//   public:
//     static OpenCLHandler *Instance();
//     bool IsInitialized() const { return initialized_; }
//     cl_context GetContext() const { return context_; }
//     cl_command_queue GetCommandQueue() const { return command_queue_; }
//     cl_device_id GetDeviceID() const { return device_id_; }

//   private:
//     OpenCLHandler();
//     ~OpenCLHandler();
//     bool initialized_;
//     cl_device_id device_id_;
//     cl_context context_;
//     cl_command_queue command_queue_;
// };

// OpenCLHandler *OpenCLHandler::Instance()
// {
//     static OpenCLHandler handler;
//     return &handler;
// }

// OpenCLHandler::OpenCLHandler() : initialized_(false)
// {
//     std::stringstream msg;
//     cl_platform_id platform_id = NULL;
//     device_id_ = NULL;
//     cl_uint ret_num_devices;
//     cl_uint ret_num_platforms;
//     cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
//     if (ret != CL_SUCCESS || ret_num_platforms == 0)
//     {
//         std::cerr << "Failed to get a valid platform" << std::endl;
//         return;
//     }

//     char buffer[1024];
//     clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1024, &buffer, NULL);
//     msg << "Found platform: " << buffer;

//     ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id_, &ret_num_devices);
//     if (ret != CL_SUCCESS)
//     {
//         std::cerr << "Failed to get valid device" << std::endl;
//         return;
//     }

//     clGetDeviceInfo(device_id_, CL_DEVICE_NAME, 1024, &buffer, NULL);
//     msg << " | Device name: " << buffer;

//     context_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, &ret);
//     command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &ret);
//     std::cout << msg.str() << std::endl;
//     initialized_ = true;
// }

// OpenCLHandler::~OpenCLHandler()
// {
//     if (initialized_)
//     {
//         cl_int ret = clFlush(command_queue_);
//         ret = clFinish(command_queue_);
//         ret = clReleaseCommandQueue(command_queue_);
//         ret = clReleaseContext(context_);
//     }
// }

// ===========================================================================
static NeuralNetwork::RMatrix MultiplyAdd(const NeuralNetwork::RMatrix &w,
                                          const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b);
static NeuralNetwork::RMatrix HadamardProduct(const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b);
static NeuralNetwork::RMatrix MatMul(const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b);

// ===========================================================================
NeuralNetwork::NeuralNetwork(std::initializer_list<int> layers_sizes) : layers_sizes_(layers_sizes)
{
    int n_layers = (int)layers_sizes_.size();
    assert(n_layers >= 3 && "There must be an input, at least one hidden, and an output layer.");

    // Random number generator to initialize the biases and weights from a normal distribution with zero mean and unit variance
    std::mt19937 gen; // (std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<Real> dist(0.0, 1.0);

    // During initializing we start from layer 1 as layer 0 is the input layer.
    for (int i = 1; i < n_layers; i++)
    {
        int sz = layers_sizes_[i];
        biases_.emplace_back(sz, 1);
        RMatrix &b = biases_.back();
        for (int j = 0; j < sz; j++)
        {
            b(j, 0) = dist(gen);
        }
    }

    for (int i = 0; i < n_layers - 1; i++)
    {
        int sz_k = layers_sizes_[i];
        int sz_j = layers_sizes_[i + 1];
        weights_.emplace_back(sz_j, sz_k);
        RMatrix &w = weights_.back();
        for (int j = 0; j < sz_j; j++)
        {
            for (int k = 0; k < sz_k; k++)
            {
                w(j, k) = dist(gen);
            }
        }
    }
}

Matrix<NeuralNetwork::Real> NeuralNetwork::FeedForward(const RMatrix &inputs)
{
    RMatrix a = inputs;
    for (int i = 0; i < biases_.size(); i++)
    {
        const RMatrix &w = weights_[i];
        const RMatrix &b = biases_[i];
        a = MultiplyAdd(w, a, b);
        ApplySigmoid(a);
    }
    return a;
}

NeuralNetwork::RMatrix MultiplyAdd(const NeuralNetwork::RMatrix &w,
                                   const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b)
{
    assert(a.getColumnsCount() == 1);
    assert(b.getColumnsCount() == 1);
    assert(w.getRowsCount() == b.getRowsCount());
    assert(a.getRowsCount() == w.getColumnsCount());

    NeuralNetwork::RMatrix result(w.getRowsCount(), 1);
    for (int i = 0; i < w.getRowsCount(); i++)
    {
        NeuralNetwork::Real dotp = 0;
        for (int k = 0; k < a.getRowsCount(); k++)
        {
            dotp += w(i, k) * a(k, 0);
        }
        result(i, 0) = dotp + b(i, 0);
    }

    return result;
}

NeuralNetwork::RMatrix HadamardProduct(const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b)
{
    assert(a.getColumnsCount() == b.getColumnsCount());
    assert(a.getRowsCount() == b.getRowsCount());
    NeuralNetwork::RMatrix p(a.getRowsCount(), a.getColumnsCount());
    for (int i = 0, count = a.getElementsCount(); i < count; i++)
    {
        p(i) = a(i) * b(i);
    }
    return p;
}

NeuralNetwork::RMatrix MatMul(const NeuralNetwork::RMatrix &a, const NeuralNetwork::RMatrix &b)
{
    assert(a.getColumnsCount() == b.getRowsCount());
    NeuralNetwork::RMatrix result(a.getRowsCount(), b.getColumnsCount());
    for (int i = 0; i < a.getRowsCount(); i++)
    {
        for (int j = 0; j < b.getColumnsCount(); j++)
        {
            result(i, j) = 0;
            for (int k = 0; k < a.getColumnsCount(); k++)
            {
                result(i, j) += (a(i, k) * b(k, j));
            }
        }
    }
    return result;
}

void NeuralNetwork::ApplySigmoid(RMatrix &zs)
{
    for (int i = 0, count = zs.getElementsCount(); i < count; i++)
    {
        zs(i) = Real(1) / (Real(1) + std::exp(-zs(i)));
    }
}

void NeuralNetwork::ApplySigmoidDeriv(RMatrix &z)
{
    ApplySigmoid(z);
    for (int i = 0, count = z.getElementsCount(); i < count; i++)
    {
        z(i) = z(i) * (Real(1) - z(i));
    }
}

void NeuralNetwork::SGD(const std::vector<RMatrix> &training_data, const std::vector<RMatrix> &training_labels,
                        int num_epochs, int mini_batch_size, Real learning_rate,
                        std::vector<RMatrix> *test_data, std::vector<RMatrix> *test_labels)
{
    std::vector<int> train_data_indices(training_data.size());
    std::iota(train_data_indices.begin(), train_data_indices.end(), 0);
    auto rng = std::default_random_engine{};
    for (int i = 0; i < num_epochs; i++)
    {
        std::shuffle(train_data_indices.begin(), train_data_indices.end(), rng);
        std::vector<RMatrix> batch_data, batch_labels;
        size_t start_idx = 0;
        size_t end_idx = mini_batch_size;
        while (true)
        {
            for (size_t k = start_idx; k < end_idx; k++)
            {
                batch_data.push_back(training_data[train_data_indices[k]]);
                batch_labels.push_back(training_labels[train_data_indices[k]]);
            }
            UpdateMiniBatch(batch_data, batch_labels, learning_rate);
            batch_data.clear();
            batch_labels.clear();
            start_idx += mini_batch_size;
            if (start_idx >= training_data.size())
            {
                break;
            }
            end_idx += mini_batch_size;
            end_idx = std::min(end_idx, training_data.size());
        }

        std::cout << "Epoch #" << i << " finished. ";
        if (test_data)
        {
            std::cout << Evaluate(*test_data, *test_labels) << "/" << test_data->size();
        }
        std::cout << std::endl;
    }
}

void NeuralNetwork::UpdateMiniBatch(const std::vector<RMatrix> &batch_data,
                                    const std::vector<RMatrix> &batch_labels, Real learning_rate)
{
    auto nabla_b = biases_;
    auto nabla_w = weights_;
    std::for_each(nabla_b.begin(), nabla_b.end(), [](RMatrix &m) { m.SetZero(); });
    std::for_each(nabla_w.begin(), nabla_w.end(), [](RMatrix &m) { m.SetZero(); });

    for (int batch_idx = 0; batch_idx < batch_data.size(); batch_idx++)
    {
        std::vector<RMatrix> delta_nabla_b, delta_nabla_w;
        BackProp(batch_data[batch_idx], batch_labels[batch_idx], delta_nabla_b, delta_nabla_w);
        for (int b_i = 0; b_i < nabla_b.size(); b_i++)
        {
            nabla_b[b_i] += delta_nabla_b[b_i];
        }

        for (int w_i = 0; w_i < nabla_w.size(); w_i++)
        {
            nabla_w[w_i] += delta_nabla_w[w_i];
        }
    }

    const Real f = learning_rate / (Real)batch_data.size();
    for (int i = 0; i < biases_.size(); i++)
    {
        auto &b = biases_[i];
        for (int k = 0, count = b.getElementsCount(); k < count; k++)
        {
            b(k) = b(k) - f * (nabla_b[i])(k);
        }
    }

    for (int i = 0; i < weights_.size(); i++)
    {
        auto &w = weights_[i];
        for (int k = 0, count = w.getElementsCount(); k < count; k++)
        {
            w(k) = w(k) - f * (nabla_w[i])(k);
        }
    }
}

void NeuralNetwork::BackProp(const RMatrix &X, const RMatrix &y,
                             std::vector<RMatrix> &nabla_b, std::vector<RMatrix> &nabla_w)
{
    nabla_b = biases_;
    nabla_w = weights_;
    std::for_each(nabla_b.begin(), nabla_b.end(), [](RMatrix &m) { m.SetZero(); });
    std::for_each(nabla_w.begin(), nabla_w.end(), [](RMatrix &m) { m.SetZero(); });

    // feedforward
    RMatrix a = X;
    std::vector<RMatrix> activations_list = {a};
    std::vector<RMatrix> zs;
    for (int i = 0; i < biases_.size(); i++)
    {
        const RMatrix &w = weights_[i];
        const RMatrix &b = biases_[i];
        a = MultiplyAdd(w, a, b);
        zs.push_back(a);
        ApplySigmoid(a);
        activations_list.push_back(a);
    }

    // backward pass
    RMatrix nabla_c = CostDeriv(activations_list.back(), y);
    RMatrix z_output = zs.back();
    ApplySigmoidDeriv(z_output);
    RMatrix delta = HadamardProduct(nabla_c, z_output);
    nabla_b.back() = delta;
    nabla_w.back() = MatMul(delta, activations_list[activations_list.size() - 2].Transpose());

    for (int i = biases_.size() - 2; i >= 0; i--)
    {
        RMatrix z = zs[i];
        ApplySigmoidDeriv(z);
        delta = HadamardProduct(MatMul(weights_[i + 1].Transpose(), delta), z);
        nabla_b[i] = delta;
        nabla_w[i] = MatMul(delta, activations_list[i].Transpose());
    }
}

NeuralNetwork::RMatrix NeuralNetwork::CostDeriv(const RMatrix &output_activations, const RMatrix &y)
{
    assert(output_activations.getRowsCount() == y.getRowsCount());
    assert(output_activations.getColumnsCount() == y.getColumnsCount() == 1);
    RMatrix result(output_activations.getRowsCount(), 1);
    for (int i = 0; i < output_activations.getElementsCount(); i++)
    {
        result(i) = output_activations(i) - y(i);
    }
    return result;
}

static int GetMaxIdx(const NeuralNetwork::RMatrix &y)
{
    assert(y.getColumnsCount() == 1);
    int idx = -1;
    NeuralNetwork::Real max_activation = std::numeric_limits<NeuralNetwork::Real>::lowest();
    for (int i = 0, count = y.getRowsCount(); i < count; i++)
    {
        if (y(i, 0) > max_activation)
        {
            max_activation = y(i, 0);
            idx = i;
        }
    }
    return idx;
}

int NeuralNetwork::Evaluate(const std::vector<RMatrix> &test_data, const std::vector<RMatrix> &test_labels)
{
    int num_correct = 0;
    for (int i = 0; i < test_data.size(); i++)
    {
        RMatrix prediction = FeedForward(test_data[i]);
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
