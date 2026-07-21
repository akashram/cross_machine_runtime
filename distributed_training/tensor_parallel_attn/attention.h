#pragma once

// Tensor-parallel multi-head attention: PLAN.md step 12. Splitting
// attention by HEAD is what makes this embarrassingly parallel: each
// head's scaled-dot-product attention (softmax(QK^T/sqrt(d))V) is
// independent of every other head, so a rank owning a subset of heads
// needs zero communication for the attention math itself. Communication
// only shows up at the boundary, in exactly the same column/row-parallel
// pattern col_row_linear/ already validated: Q/K/V projections are
// column-parallel (split by output columns = which heads a rank owns; no
// forward communication), the output projection is row-parallel (split by
// input rows; one all-reduce SUM in forward, one for dX in backward).
//
// Simplified vs. real Megatron for clarity under time constraints: Q, K,
// V use three separate weight matrices rather than one fused QKV matrix
// (mathematically equivalent — three column-parallel projections instead
// of one wider one split three ways internally), and there is no
// causal/padding mask (irrelevant to what is being validated here:
// gradient correctness of the tensor-parallel split, not attention
// masking, which is orthogonal and unaffected by sharding).

#include <cmath>

#include "matrix.h"

namespace distributed_training {

inline Matrix softmax_rows(const Matrix &s) {
  Matrix out(s.rows(), s.cols());
  for (int i = 0; i < s.rows(); ++i) {
    float max_val = s(i, 0);
    for (int j = 1; j < s.cols(); ++j) max_val = std::max(max_val, s(i, j));
    float denom = 0.0f;
    for (int j = 0; j < s.cols(); ++j) denom += std::exp(s(i, j) - max_val);
    for (int j = 0; j < s.cols(); ++j) out(i, j) = std::exp(s(i, j) - max_val) / denom;
  }
  return out;
}

// Standard softmax Jacobian-vector product, applied row-wise:
// dS_i = A_i * (dA_i - sum_j A_ij * dA_ij).
inline Matrix softmax_rows_backward(const Matrix &a, const Matrix &da) {
  Matrix ds(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i) {
    float dot = 0.0f;
    for (int j = 0; j < a.cols(); ++j) dot += a(i, j) * da(i, j);
    for (int j = 0; j < a.cols(); ++j) ds(i, j) = a(i, j) * (da(i, j) - dot);
  }
  return ds;
}

struct AttentionCache {
  Matrix q, k, v, a; // saved for backward
};

inline Matrix single_head_attention_forward(const Matrix &q, const Matrix &k, const Matrix &v, AttentionCache &cache) {
  float scale = 1.0f / std::sqrt(static_cast<float>(q.cols()));
  Matrix s = q.matmul(k.transpose()) * scale;
  Matrix a = softmax_rows(s);
  cache = AttentionCache{q, k, v, a};
  return a.matmul(v);
}

struct AttentionGrads {
  Matrix dq, dk, dv;
};

inline AttentionGrads single_head_attention_backward(const AttentionCache &cache, const Matrix &d_out) {
  float scale = 1.0f / std::sqrt(static_cast<float>(cache.q.cols()));
  Matrix dv = cache.a.transpose().matmul(d_out);
  Matrix da = d_out.matmul(cache.v.transpose());
  Matrix ds = softmax_rows_backward(cache.a, da) * scale;
  Matrix dq = ds.matmul(cache.k);
  Matrix dk = ds.transpose().matmul(cache.q);
  return AttentionGrads{dq, dk, dv};
}

} // namespace distributed_training
