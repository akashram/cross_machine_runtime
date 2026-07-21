#pragma once

// Minimal row-major dense float matrix — the value type the autograd tape
// (autograd.h) operates on. Deliberately not foundation::TensorHandle:
// TensorHandle is a memory descriptor (shape/stride/dtype, zero-copy
// views) with no arithmetic on it at all; a reverse-mode tape needs
// concrete elementwise/matmul ops on 2-D batches, which is simpler to give
// value semantics here than to layer on top of TensorHandle's view
// machinery for no benefit at this scale.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace distributed_training {

class Matrix {
public:
  Matrix() = default;
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f) {}

  static Matrix zeros(int rows, int cols) { return Matrix(rows, cols); }

  static Matrix random(int rows, int cols, std::mt19937 &rng, float stddev) {
    Matrix m(rows, cols);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (float &v : m.data_) v = dist(rng);
    return m;
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  size_t size() const { return data_.size(); }

  float &operator()(int r, int c) {
    assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
    return data_[static_cast<size_t>(r) * static_cast<size_t>(cols_) + static_cast<size_t>(c)];
  }
  float operator()(int r, int c) const {
    assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
    return data_[static_cast<size_t>(r) * static_cast<size_t>(cols_) + static_cast<size_t>(c)];
  }

  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }

  void fill(float v) { std::fill(data_.begin(), data_.end(), v); }

  Matrix transpose() const {
    Matrix out(cols_, rows_);
    for (int r = 0; r < rows_; ++r)
      for (int c = 0; c < cols_; ++c) out(c, r) = (*this)(r, c);
    return out;
  }

  Matrix matmul(const Matrix &other) const {
    assert(cols_ == other.rows_);
    Matrix out(rows_, other.cols_);
    for (int i = 0; i < rows_; ++i) {
      for (int k = 0; k < cols_; ++k) {
        float a = (*this)(i, k);
        if (a == 0.0f) continue;
        for (int j = 0; j < other.cols_; ++j) out(i, j) += a * other(k, j);
      }
    }
    return out;
  }

  Matrix operator+(const Matrix &other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    Matrix out(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] + other.data_[i];
    return out;
  }

  Matrix operator-(const Matrix &other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    Matrix out(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] - other.data_[i];
    return out;
  }

  Matrix elementwise_mul(const Matrix &other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    Matrix out(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] * other.data_[i];
    return out;
  }

  Matrix operator*(float scalar) const {
    Matrix out(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] * scalar;
    return out;
  }

  // Broadcast-adds a [1 x cols] row vector to every row (bias add).
  Matrix add_row_broadcast(const Matrix &row) const {
    assert(row.rows_ == 1 && row.cols_ == cols_);
    Matrix out(rows_, cols_);
    for (int r = 0; r < rows_; ++r)
      for (int c = 0; c < cols_; ++c) out(r, c) = (*this)(r, c) + row(0, c);
    return out;
  }

  // Sums each column down to a [1 x cols] row vector — the adjoint of
  // add_row_broadcast, used in bias-gradient backward.
  Matrix sum_rows() const {
    Matrix out(1, cols_);
    for (int r = 0; r < rows_; ++r)
      for (int c = 0; c < cols_; ++c) out(0, c) += (*this)(r, c);
    return out;
  }

  float sum() const {
    float s = 0.0f;
    for (float v : data_) s += v;
    return s;
  }

  template <typename Fn>
  Matrix apply(Fn fn) const {
    Matrix out(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = fn(data_[i]);
    return out;
  }

  void add_inplace(const Matrix &other, float scale = 1.0f) {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i] * scale;
  }

private:
  int rows_ = 0;
  int cols_ = 0;
  std::vector<float> data_;
};

} // namespace distributed_training
