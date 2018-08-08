#pragma once

#include <assert.h>
#include <initializer_list>
#include <vector>
#include <algorithm>

template <typename T>
class Matrix
{
  public:
    Matrix(int nrows, int ncols);
    ~Matrix();
    Matrix(const Matrix &);
    Matrix &operator=(const Matrix &);
    Matrix(Matrix &&);
    Matrix &operator=(Matrix &&);

    void SetZero();
    Matrix<T> Transpose() const;

    T &operator()(int row, int col);
    const T &operator()(int row, int col) const;
    T &operator()(int idx);
    const T &operator()(int idx) const;

    Matrix<T> &operator+=(const Matrix<T> &);

    int getElementsCount() const;
    int getRowsCount() const;
    int getColumnsCount() const;

    T *GetDataPtr() const { return data_; } // This function is made const only so I can easily get the pointer on const objects without having to const_cast everywhere.

  private:
    void allocate_storage(size_t n_elements);
    void free_storage();
    int nrows_, ncols_;
    T *data_;
};

//===============================================
class NeuralNetwork
{
  public:
    using Real = float;
    using RMatrix = Matrix<Real>;

    NeuralNetwork(std::initializer_list<int> layers_sizes);

    RMatrix FeedForward(const RMatrix &inputs);

    void SGD(const std::vector<RMatrix> &training_data, const std::vector<RMatrix> &training_labels,
             int num_epochs, int mini_batch_size, Real learning_rate,
             std::vector<RMatrix> *test_data = nullptr, std::vector<RMatrix> *test_labels = nullptr);

  private:
    void UpdateMiniBatch(const std::vector<RMatrix> &batch_data, const std::vector<RMatrix> &batch_labels, Real learning_rate);
    void BackProp(const RMatrix &X, const RMatrix &y, std::vector<RMatrix> &nabla_b, std::vector<RMatrix> &nabla_w);
    int Evaluate(const std::vector<RMatrix> &test_data, const std::vector<RMatrix> &test_labels); // returns the number of correct predictions
    RMatrix CostDeriv(const RMatrix &activations, const RMatrix &y);
    void ApplySigmoid(RMatrix &z);
    void ApplySigmoidDeriv(RMatrix &z);

    std::vector<int> layers_sizes_;
    std::vector<RMatrix> biases_;
    std::vector<RMatrix> weights_;
};

//////////////////////////////////////////////
// Matrix Implementation
//////////////////////////////////////////////

template <typename T>
Matrix<T>::Matrix(int nrows, int ncols) : nrows_(nrows), ncols_(ncols), data_(nullptr)
{
    allocate_storage(nrows_ * ncols_);
}

template <typename T>
Matrix<T>::~Matrix()
{
    if (data_ != nullptr)
    {
        free_storage();
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &m) : nrows_(m.nrows_), ncols_(m.ncols_), data_(nullptr)
{
    allocate_storage(nrows_ * ncols_);
    std::copy(m.data_, m.data_ + (nrows_ * ncols_), data_);
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &m)
{
    if (this != &m)
    {
        int current_n_elements = nrows_ * ncols_;
        int new_n_elements = m.nrows_ * m.ncols_;
        if (current_n_elements != new_n_elements)
        {
            free_storage();
            allocate_storage(new_n_elements);
        }
        nrows_ = m.nrows_;
        ncols_ = m.ncols_;
        std::copy(m.data_, m.data_ + new_n_elements, data_);
    }
    return *this;
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&m) : nrows_(0), ncols_(0), data_(nullptr)
{
    nrows_ = m.nrows_;
    ncols_ = m.ncols_;
    data_ = m.data_;
    m.ncols_ = 0;
    m.nrows_ = 0;
    m.data_ = nullptr;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&m)
{
    if (this != &m)
    {
        free_storage();
        nrows_ = m.nrows_;
        ncols_ = m.ncols_;
        data_ = m.data_;
        m.ncols_ = 0;
        m.nrows_ = 0;
        m.data_ = nullptr;
    }
    return *this;
}

template <typename T>
void Matrix<T>::SetZero()
{
    std::fill(data_, data_ + (nrows_ * ncols_), (T)0);
}

template <typename T>
Matrix<T> Matrix<T>::Transpose() const
{
    Matrix<T> t(ncols_, nrows_);
    for (int i = 0; i < ncols_; i++)
    {
        for (int j = 0; j < nrows_; j++)
        {
            t(i, j) = (*this)(j, i);
        }
    }
    return t;
}

template <typename T>
T &Matrix<T>::operator()(int row, int col)
{
    assert(row >= 0 && row < getRowsCount());
    assert(col >= 0 && col < getColumnsCount());
    return data_[ncols_ * row + col];
}

template <typename T>
const T &Matrix<T>::operator()(int row, int col) const
{
    assert(row >= 0 && row < getRowsCount());
    assert(col >= 0 && col < getColumnsCount());
    return data_[ncols_ * row + col];
}

template <typename T>
T &Matrix<T>::operator()(int idx)
{
    assert(idx >= 0 && idx < getElementsCount());
    return data_[idx];
}

template <typename T>
const T &Matrix<T>::operator()(int idx) const
{
    assert(idx >= 0 && idx < getElementsCount());
    return data_[idx];
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &m)
{
    assert(nrows_ == m.nrows_);
    assert(ncols_ == m.ncols_);
    for (int i = 0, count = nrows_ * ncols_; i < count; i++)
    {
        data_[i] += m.data_[i];
    }
    return *this;
}

template <typename T>
int Matrix<T>::getElementsCount() const
{
    return (nrows_ * ncols_);
}

template <typename T>
int Matrix<T>::getRowsCount() const
{
    return nrows_;
}

template <typename T>
int Matrix<T>::getColumnsCount() const
{
    return ncols_;
}

template <typename T>
void Matrix<T>::allocate_storage(size_t n_elements)
{
    assert(data_ == nullptr);
    data_ = (T *)malloc(sizeof(T) * n_elements);
}

template <typename T>
void Matrix<T>::free_storage()
{
    assert(data_ != nullptr);
    free((void *)data_);
    data_ = nullptr;
}