// tensor_parallel_linear_test.cpp — a single-process reference (full,
// unsharded weights, hand-derived relu-MLP forward/backward) vs. 4
// simulated tensor-parallel ranks running the column-then-row-parallel
// fused block, real communication (ring_allreduce for both the forward
// output all-reduce and the backward dX all-reduce). Compares forward
// output, every weight/bias gradient (reassembled from shards), and the
// input gradient dX.
#include "tensor_parallel_linear.h"

#include "ring_allreduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

Matrix relu_forward(const Matrix &pre) {
  return pre.apply([](float v) { return v > 0.0f ? v : 0.0f; });
}

Matrix relu_backward(const Matrix &pre, const Matrix &dpost) {
  Matrix mask = pre.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; });
  return dpost.elementwise_mul(mask);
}

// Column-slice [start, start+width) of every row.
Matrix col_slice(const Matrix &m, int start, int width) {
  Matrix out(m.rows(), width);
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < width; ++j) out(i, j) = m(i, start + j);
  return out;
}

// Row-slice [start, start+height).
Matrix row_slice(const Matrix &m, int start, int height) {
  Matrix out(height, m.cols());
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}

float max_abs_diff(const Matrix &a, const Matrix &b) {
  float max_diff = 0.0f;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j) max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
  return max_diff;
}

} // namespace

int main() {
  constexpr int kBatch = 6;
  constexpr int kIn = 8;
  constexpr int kHidden = 16;
  constexpr int kOut = 8;
  constexpr int kWorldSize = 4;
  constexpr int kHiddenShard = kHidden / kWorldSize; // 4
  constexpr uint16_t kBasePort = 35451;

  std::mt19937 rng(55);
  std::normal_distribution<float> dist(0.0f, 0.3f);

  Matrix x(kBatch, kIn);
  for (int i = 0; i < kBatch; ++i)
    for (int j = 0; j < kIn; ++j) x(i, j) = dist(rng);

  Matrix w_col(kIn, kHidden), b_col(1, kHidden), w_row(kHidden, kOut), row_bias(1, kOut);
  for (int i = 0; i < kIn; ++i)
    for (int j = 0; j < kHidden; ++j) w_col(i, j) = dist(rng);
  for (int j = 0; j < kHidden; ++j) b_col(0, j) = dist(rng);
  for (int i = 0; i < kHidden; ++i)
    for (int j = 0; j < kOut; ++j) w_row(i, j) = dist(rng);
  for (int j = 0; j < kOut; ++j) row_bias(0, j) = dist(rng);

  // --- Single-process reference: plain 2-layer relu MLP, loss = 0.5*sum(y^2).
  Matrix y1_pre_ref = x.matmul(w_col).add_row_broadcast(b_col);
  Matrix y1_ref = relu_forward(y1_pre_ref);
  Matrix y_ref = y1_ref.matmul(w_row).add_row_broadcast(row_bias);

  Matrix dy_ref = y_ref; // d(0.5*sum(y^2))/dy = y
  Matrix drow_bias_ref = dy_ref.sum_rows();
  Matrix dw_row_ref = y1_ref.transpose().matmul(dy_ref);
  Matrix dy1_ref = dy_ref.matmul(w_row.transpose());
  Matrix dy1_pre_ref = relu_backward(y1_pre_ref, dy1_ref);
  Matrix dw_col_ref = x.transpose().matmul(dy1_pre_ref);
  Matrix db_col_ref = dy1_pre_ref.sum_rows();
  Matrix dx_ref = dy1_pre_ref.matmul(w_col.transpose());

  // --- Tensor-parallel: 4 ranks, real communication.
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  struct RankResult {
    Matrix y, dw_col_shard, db_col_shard, dw_row_shard, drow_bias, dx;
  };
  std::vector<std::future<RankResult>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> RankResult {
      ColumnParallelLinear col{col_slice(w_col, r * kHiddenShard, kHiddenShard), col_slice(b_col, r * kHiddenShard, kHiddenShard)};
      RowParallelLinear row{row_slice(w_row, r * kHiddenShard, kHiddenShard)};

      Matrix y1_pre_shard = col.forward(x);
      Matrix y1_shard = relu_forward(y1_pre_shard);
      Matrix y_partial = row.forward(y1_shard);

      ring_allreduce(y_partial.data(), y_partial.size(), *ch); // now the full sum on every rank
      Matrix y = y_partial.add_row_broadcast(row_bias);

      Matrix dy = y; // same loss as the reference, so same dy formula
      Matrix drow_bias = dy.sum_rows(); // replicated bias -> every rank computes the same value redundantly, no comm needed

      auto row_grads = row.backward(y1_shard, dy);
      Matrix dy1_shard_post = row_grads.dx_shard;
      Matrix dy1_pre_shard = relu_backward(y1_pre_shard, dy1_shard_post);

      auto col_grads = col.backward(x, dy1_pre_shard);
      Matrix dx_partial = col_grads.dx_partial;
      ring_allreduce(dx_partial.data(), dx_partial.size(), *ch); // now the full dX on every rank

      return RankResult{y, col_grads.dweight, col_grads.dbias, row_grads.dweight, drow_bias, dx_partial};
    }));
  }

  std::vector<RankResult> rank_results;
  for (auto &f : results) rank_results.push_back(f.get());

  // Reassemble sharded gradients into full matrices for comparison.
  Matrix dw_col_reassembled(kIn, kHidden);
  Matrix db_col_reassembled(1, kHidden);
  Matrix dw_row_reassembled(kHidden, kOut);
  for (int r = 0; r < kWorldSize; ++r) {
    for (int i = 0; i < kIn; ++i)
      for (int j = 0; j < kHiddenShard; ++j) dw_col_reassembled(i, r * kHiddenShard + j) = rank_results[static_cast<size_t>(r)].dw_col_shard(i, j);
    for (int j = 0; j < kHiddenShard; ++j) db_col_reassembled(0, r * kHiddenShard + j) = rank_results[static_cast<size_t>(r)].db_col_shard(0, j);
    for (int i = 0; i < kHiddenShard; ++i)
      for (int j = 0; j < kOut; ++j) dw_row_reassembled(r * kHiddenShard + i, j) = rank_results[static_cast<size_t>(r)].dw_row_shard(i, j);
  }

  bool ok = true;
  auto check = [&](const char *name, float diff, float tol = 1e-3f) {
    std::printf("  %-24s max abs diff = %.6f: %s\n", name, diff, diff <= tol ? "PASS" : "FAIL");
    if (diff > tol) ok = false;
  };

  check("forward y (rank 0)", max_abs_diff(y_ref, rank_results[0].y));
  check("dW_col (reassembled)", max_abs_diff(dw_col_ref, dw_col_reassembled));
  check("db_col (reassembled)", max_abs_diff(db_col_ref, db_col_reassembled));
  check("dW_row (reassembled)", max_abs_diff(dw_row_ref, dw_row_reassembled));
  check("d(row_bias) (rank 0)", max_abs_diff(drow_bias_ref, rank_results[0].drow_bias));
  check("dX (rank 0, all-reduced)", max_abs_diff(dx_ref, rank_results[0].dx));

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
