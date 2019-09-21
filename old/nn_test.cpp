#include "neural_network.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdint>

static void plot(const NeuralNetwork::RMatrix &X)
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            NeuralNetwork::Real pixel_val = X(i * 28 + j, 0);
            if (pixel_val < 1)
            {
                std::cout << " ";
            }
            else
            {
                std::cout << "*";
            }
        }
        std::cout << std::endl;
    }
}

static bool read_imgs_file(std::string file_name, std::vector<NeuralNetwork::RMatrix> &imgs)
{
    std::ifstream imgs_input(file_name, std::ios::binary);
    if (!imgs_input.is_open())
    {
        return false;
    }

    std::vector<char> imgs_buffer((
                                      std::istreambuf_iterator<char>(imgs_input)),
                                  (std::istreambuf_iterator<char>()));

    size_t idx = 16; // image data starts at byte 16
    const size_t sz = imgs_buffer.size();
    const size_t img_sz = 28 * 28;
    NeuralNetwork::Real min = std::numeric_limits<NeuralNetwork::Real>::max();
    NeuralNetwork::Real max = std::numeric_limits<NeuralNetwork::Real>::min();
    while (idx < sz)
    {
        NeuralNetwork::RMatrix img(img_sz, 1);
        for (int bi = 0; bi < img_sz; bi++)
        {
            img(bi, 0) = static_cast<NeuralNetwork::Real>(static_cast<unsigned char>(imgs_buffer[idx + bi]));
            if (img(bi, 0) < min)
            {
                min = img(bi, 0);
            }
            if (img(bi, 0) > max)
            {
                max = img(bi, 0);
            }
        }
        imgs.emplace_back(std::move(img));
        idx += img_sz;
    }

    // Normalize
    const auto range = max - min;
    if (range == 0.0f)
    {
        return true;
    }

    for (auto &img : imgs)
    {
        for (size_t i = 0; i < img_sz; i++)
        {
            img(i, 0) /= range;
        }
    }

    return true;
}

static bool read_labels_file(std::string file_name, std::vector<NeuralNetwork::RMatrix> &labels)
{
    std::ifstream labels_input(file_name, std::ios::binary);
    if (!labels_input.is_open())
    {
        return false;
    }

    std::vector<char> labels_buffer((
                                        std::istreambuf_iterator<char>(labels_input)),
                                    (std::istreambuf_iterator<char>()));

    size_t idx = 8; // labels data starts at byte 8
    for (size_t i = idx, sz = labels_buffer.size(); i < sz; i++)
    {
        unsigned char label_val = static_cast<unsigned char>(labels_buffer[i]);
        NeuralNetwork::RMatrix label_mtrx(10, 1);
        label_mtrx.SetZero();
        label_mtrx(label_val, 0) = static_cast<NeuralNetwork::Real>(1);
        labels.emplace_back(std::move(label_mtrx));
    }

    return true;
}

int main()
{
    std::vector<NeuralNetwork::RMatrix> train_imgs, train_labels, validation_imgs, validation_labels;
    if (!read_imgs_file("train-images.idx3-ubyte", train_imgs) || !read_labels_file("train-labels.idx1-ubyte", train_labels))
    {
        std::cout << "Unable to read training and testing data files" << std::endl;
        return -1;
    }

    validation_imgs.reserve(10000);
    validation_labels.reserve(10000);
    for (int i = train_imgs.size() - 1; i >= 50000; i--)
    {
        validation_imgs.emplace_back(std::move(train_imgs.back()));
        train_imgs.pop_back();
        validation_labels.emplace_back(std::move(train_labels.back()));
        train_labels.pop_back();
    }

    NeuralNetwork nn = {784, 30, 10};
    nn.SGD(train_imgs, train_labels, 30, 10, 3.0, &validation_imgs, &validation_labels);
    return 0;
}
