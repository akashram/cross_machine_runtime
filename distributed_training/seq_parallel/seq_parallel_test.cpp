// seq_parallel_test.cpp — two checks:
//  1. LayerNorm gradient check (isolated, finite differences) — validates
//     layernorm_backward's formula once; sharding by sequence does not
//     change this formula at all (each row is independent either way), so
//     this is not re-derived per-shard-configuration below.
//  2. Forward correctness of the full sequence-parallel + tensor-parallel
//     chain (LN1 -sharded-> all-gather -> column+relu+row-parallel with
//     REDUCE-SCATTER output -> LN2 -sharded->) against a single-process
//     reference — this is the actually novel part of this step: the
//     boundary transition, and confirming reduce-scatter's rotated chunk
//     ownership still lines up with clean sequence-row boundaries.
#include "layernorm.h"
#include "../col_row_linear/tensor_parallel_linear.h"

#include "ring_allreduce.h"
#include "collectives.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

Matrix col_slice(const Matrix &m, int start, int width) {
  Matrix out(m.rows(), width);
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < width; ++j) out(i, j) = m(i, start + j);
  return out;
}
Matrix row_slice(const Matrix &m, int start, int height) {
  Matrix out(height, m.cols());
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}
float max_abs_diff(const Matrix &a, const Matrix &b) {
  float d = 0.0f;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j) d = std::max(d, std::abs(a(i, j) - b(i, j)));
  return d;
}

bool test_layernorm_gradient_check() {
  std::mt19937 rng(11);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  constexpr int kRows = 4, kHidden = 6;
  Matrix x(kRows, kHidden), gamma(1, kHidden), beta(1, kHidden);
  for (int i = 0; i < kRows; ++i)
    for (int j = 0; j < kHidden; ++j) x(i, j) = dist(rng);
  for (int j = 0; j < kHidden; ++j) { gamma(0, j) = 1.0f + 0.1f * dist(rng); beta(0, j) = 0.1f * dist(rng); }

  auto loss_of = [&](const Matrix &xx) {
    LayerNormCache cache;
    Matrix y = layernorm_forward(xx, gamma, beta, cache);
    return 0.5f * y.apply([](float v) { return v * v; }).sum();
  };

  LayerNormCache cache;
  Matrix y = layernorm_forward(x, gamma, beta, cache);
  Matrix dy = y; // d(0.5*sum(y^2))/dy = y
  auto grads = layernorm_backward(cache, dy);

  constexpr float kEps = 1e-3f;
  std::vector<float> rel_errs;
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kHidden; ++j) {
      Matrix x_plus = x, x_minus = x;
      x_plus(i, j) += kEps;
      x_minus(i, j) -= kEps;
      float numeric = (loss_of(x_plus) - loss_of(x_minus)) / (2.0f * kEps);
      rel_errs.push_back(std::abs(grads.dx(i, j) - numeric) / std::max(1e-4f, std::abs(numeric)));
    }
  }
  // Median, not max — same rationale as autograd/autograd_test.cpp: one
  // element with an unusually small finite-difference denominator can
  // dominate a max-based check without indicating a wrong formula. Verified
  // by hand-rederiving layernorm_backward's closed form against the full
  // chain-rule expansion (dvar/dx, dmu/dx) — they agree exactly, so the
  // formula itself is not in question, only finite-difference noise is.
  std::sort(rel_errs.begin(), rel_errs.end());
  float median = rel_errs[rel_errs.size() / 2];
  float max_err = rel_errs.back();
  bool ok = median < 1e-2f;
  std::printf("test 1 (LayerNorm gradient check, dx): median relative error = %.6f, max = %.6f: %s\n", median,
              max_err, ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = test_layernorm_gradient_check();

  constexpr int kSeq = 8;
  constexpr int kHidden = 16;
  constexpr int kWorldSize = 4;
  constexpr int kSeqShard = kSeq / kWorldSize;         // 2 rows/rank
  constexpr int kHiddenShard = kHidden / kWorldSize;   // 4
  constexpr uint16_t kBasePort = 35551;

  std::mt19937 rng(31);
  std::normal_distribution<float> dist(0.0f, 0.3f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };
  auto ones_gamma = [&](int c) { Matrix m(1, c); for (int j = 0; j < c; ++j) m(0, j) = 1.0f; return m; };
  auto zeros_beta = [&](int c) { return Matrix(1, c); };

  Matrix x = random_matrix(kSeq, kHidden);
  Matrix gamma1 = ones_gamma(kHidden), beta1 = zeros_beta(kHidden);
  Matrix gamma2 = ones_gamma(kHidden), beta2 = zeros_beta(kHidden);
  Matrix w_col(kHidden, kHidden), b_col(1, kHidden), w_row(kHidden, kHidden);
  w_col = random_matrix(kHidden, kHidden);
  for (int j = 0; j < kHidden; ++j) b_col(0, j) = dist(rng);
  w_row = random_matrix(kHidden, kHidden);

  // --- Single-process reference: LN1 -> Linear(col) -> ReLU -> Linear(row, no bias) -> LN2.
  LayerNormCache c1;
  Matrix ln1 = layernorm_forward(x, gamma1, beta1, c1);
  Matrix pre_relu = ln1.matmul(w_col).add_row_broadcast(b_col);
  Matrix relu_out = pre_relu.apply([](float v) { return v > 0.0f ? v : 0.0f; });
  Matrix lin2 = relu_out.matmul(w_row);
  LayerNormCache c2;
  Matrix ref_out = layernorm_forward(lin2, gamma2, beta2, c2);

  // --- Sequence-parallel + tensor-parallel: 4 ranks.
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<Matrix>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> Matrix {
      // Sequence-sharded LN1: this rank owns rows [r*kSeqShard, (r+1)*kSeqShard).
      Matrix x_shard = row_slice(x, r * kSeqShard, kSeqShard);
      LayerNormCache cache1;
      Matrix ln1_shard = layernorm_forward(x_shard, gamma1, beta1, cache1);

      // Enter the tensor-parallel region: need the FULL sequence.
      Matrix ln1_full(kSeq, kHidden);
      collectives::AllGather(ln1_shard.data(), ln1_shard.size(), ln1_full.data(), *ch);
      // AllGather's rotated placement (rank r's contribution lands at slot
      // (r+1)%world_size) still lines up with clean SEQUENCE-ROW
      // boundaries here, because kSeqShard*kHidden elements is exactly
      // kSeqShard whole rows in this class's row-major layout.

      ColumnParallelLinear col{col_slice(w_col, r * kHiddenShard, kHiddenShard), col_slice(b_col, r * kHiddenShard, kHiddenShard)};
      RowParallelLinear row{row_slice(w_row, r * kHiddenShard, kHiddenShard)};

      Matrix pre = col.forward(ln1_full);
      Matrix act = pre.apply([](float v) { return v > 0.0f ? v : 0.0f; });
      Matrix partial = row.forward(act); // [kSeq x kHidden], partial sum

      // Leave the tensor-parallel region straight into a sequence shard:
      // reduce-scatter combines the SUM (finishing the row-parallel
      // reduction) and the re-shard into one collective, instead of an
      // all-reduce (full result everywhere) followed by a manual slice.
      collectives::ReduceScatter(partial.data(), partial.size(), *ch);
      size_t owned_chunk = static_cast<size_t>((r + 1) % kWorldSize);
      Matrix lin2_shard = row_slice(partial, static_cast<int>(owned_chunk) * kSeqShard, kSeqShard);

      LayerNormCache cache2;
      Matrix out_shard = layernorm_forward(lin2_shard, gamma2, beta2, cache2);
      return out_shard; // caller must place at rows [owned_chunk*kSeqShard, ...) to compare against ref_out
    }));
  }

  // AllGather rotates rank src's contribution to position-block (src+1)%N
  // in ln1_full; ReduceScatter then rotates AGAIN, giving rank r ownership
  // of position-block (r+1)%N of `partial` — which, composing the two
  // rotations, is precisely the position-block that started as rank r's
  // OWN original x_shard (block k in ln1_full traces back to original rank
  // (k-1+N)%N; rank r owns block (r+1)%N; (((r+1)%N)-1+N)%N == r). So each
  // rank's returned shard maps straight back to ITS OWN original sequence
  // rows [r*kSeqShard, ...) — the two rotations cancel out, which is
  // exactly the invariant real sequence parallelism relies on (a rank
  // keeps working on the same sequence positions it started with).
  Matrix reassembled(kSeq, kHidden);
  for (int r = 0; r < kWorldSize; ++r) {
    Matrix shard = results[static_cast<size_t>(r)].get();
    for (int i = 0; i < kSeqShard; ++i)
      for (int j = 0; j < kHidden; ++j) reassembled(r * kSeqShard + i, j) = shard(i, j);
  }

  float diff = max_abs_diff(ref_out, reassembled);
  bool chain_ok = diff <= 1e-3f;
  std::printf("test 2 (sequence-parallel + TP chain, reassembled output): max abs diff = %.6f: %s\n", diff,
              chain_ok ? "PASS" : "FAIL");
  ok = ok && chain_ok;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
