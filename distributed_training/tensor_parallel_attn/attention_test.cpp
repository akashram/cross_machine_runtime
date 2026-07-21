// attention_test.cpp — single-process reference (full Wq/Wk/Wv/Wout,
// all-heads attention, hand-derived backward) vs. 4 simulated
// tensor-parallel ranks each owning a subset of heads, real communication
// (ring_allreduce for the row-parallel output projection's forward
// all-reduce and the column-parallel projections' backward dX all-reduce).
#include "attention.h"

#include "ring_allreduce.h"

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

// Full (unsharded), all-heads attention forward + hand-derived backward —
// the single-process reference. Loops over heads and concatenates,
// mirroring what the tensor-parallel version does per-rank over its
// OWNED subset of heads.
struct FullAttentionResult {
  Matrix out;
  Matrix dwq, dwk, dwv, dwout, dx;
};

FullAttentionResult run_reference(const Matrix &x, const Matrix &wq, const Matrix &wk, const Matrix &wv,
                                   const Matrix &wout, int num_heads, int head_dim) {
  Matrix q_full = x.matmul(wq), k_full = x.matmul(wk), v_full = x.matmul(wv);
  int seq = x.rows();
  Matrix o_full(seq, num_heads * head_dim);
  std::vector<AttentionCache> caches(static_cast<size_t>(num_heads));
  for (int h = 0; h < num_heads; ++h) {
    Matrix qh = col_slice(q_full, h * head_dim, head_dim);
    Matrix kh = col_slice(k_full, h * head_dim, head_dim);
    Matrix vh = col_slice(v_full, h * head_dim, head_dim);
    Matrix oh = single_head_attention_forward(qh, kh, vh, caches[static_cast<size_t>(h)]);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) o_full(i, h * head_dim + j) = oh(i, j);
  }
  Matrix out = o_full.matmul(wout);

  Matrix d_out = out; // loss = 0.5*sum(out^2)
  Matrix dwout = o_full.transpose().matmul(d_out);
  Matrix do_full = d_out.matmul(wout.transpose());

  Matrix dq_full(seq, num_heads * head_dim), dk_full(seq, num_heads * head_dim), dv_full(seq, num_heads * head_dim);
  for (int h = 0; h < num_heads; ++h) {
    Matrix doh = col_slice(do_full, h * head_dim, head_dim);
    auto g = single_head_attention_backward(caches[static_cast<size_t>(h)], doh);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) {
        dq_full(i, h * head_dim + j) = g.dq(i, j);
        dk_full(i, h * head_dim + j) = g.dk(i, j);
        dv_full(i, h * head_dim + j) = g.dv(i, j);
      }
  }
  Matrix dwq = x.transpose().matmul(dq_full);
  Matrix dwk = x.transpose().matmul(dk_full);
  Matrix dwv = x.transpose().matmul(dv_full);
  Matrix dx = dq_full.matmul(wq.transpose()) + dk_full.matmul(wk.transpose()) + dv_full.matmul(wv.transpose());

  return FullAttentionResult{out, dwq, dwk, dwv, dwout, dx};
}

} // namespace

int main() {
  constexpr int kSeq = 5;
  constexpr int kHidden = 16;
  constexpr int kNumHeads = 8;
  constexpr int kHeadDim = kHidden / kNumHeads; // 2
  constexpr int kWorldSize = 4;
  constexpr int kHiddenShard = kHidden / kWorldSize;      // 4
  constexpr int kHeadsPerRank = kHiddenShard / kHeadDim;  // 2
  constexpr uint16_t kBasePort = 35501;

  std::mt19937 rng(21);
  std::normal_distribution<float> dist(0.0f, 0.3f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };

  Matrix x = random_matrix(kSeq, kHidden);
  Matrix wq = random_matrix(kHidden, kHidden);
  Matrix wk = random_matrix(kHidden, kHidden);
  Matrix wv = random_matrix(kHidden, kHidden);
  Matrix wout = random_matrix(kHidden, kHidden);

  auto ref = run_reference(x, wq, wk, wv, wout, kNumHeads, kHeadDim);

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  struct RankResult {
    Matrix out, dwq_shard, dwk_shard, dwv_shard, dwout_shard, dx;
  };
  std::vector<std::future<RankResult>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> RankResult {
      Matrix wq_shard = col_slice(wq, r * kHiddenShard, kHiddenShard);
      Matrix wk_shard = col_slice(wk, r * kHiddenShard, kHiddenShard);
      Matrix wv_shard = col_slice(wv, r * kHiddenShard, kHiddenShard);
      Matrix wout_shard = row_slice(wout, r * kHiddenShard, kHiddenShard);

      Matrix q = x.matmul(wq_shard), k = x.matmul(wk_shard), v = x.matmul(wv_shard);
      Matrix o_local(kSeq, kHiddenShard);
      std::vector<AttentionCache> caches(static_cast<size_t>(kHeadsPerRank));
      for (int h = 0; h < kHeadsPerRank; ++h) {
        Matrix qh = col_slice(q, h * kHeadDim, kHeadDim);
        Matrix kh = col_slice(k, h * kHeadDim, kHeadDim);
        Matrix vh = col_slice(v, h * kHeadDim, kHeadDim);
        Matrix oh = single_head_attention_forward(qh, kh, vh, caches[static_cast<size_t>(h)]);
        for (int i = 0; i < kSeq; ++i)
          for (int j = 0; j < kHeadDim; ++j) o_local(i, h * kHeadDim + j) = oh(i, j);
      }

      Matrix out_partial = o_local.matmul(wout_shard);
      ring_allreduce(out_partial.data(), out_partial.size(), *ch);
      Matrix out = out_partial; // now the full, correct output on every rank

      Matrix d_out = out;
      Matrix dwout_shard = o_local.transpose().matmul(d_out);
      Matrix do_local = d_out.matmul(wout_shard.transpose());

      Matrix dq(kSeq, kHiddenShard), dk(kSeq, kHiddenShard), dv(kSeq, kHiddenShard);
      for (int h = 0; h < kHeadsPerRank; ++h) {
        Matrix doh = col_slice(do_local, h * kHeadDim, kHeadDim);
        auto g = single_head_attention_backward(caches[static_cast<size_t>(h)], doh);
        for (int i = 0; i < kSeq; ++i)
          for (int j = 0; j < kHeadDim; ++j) {
            dq(i, h * kHeadDim + j) = g.dq(i, j);
            dk(i, h * kHeadDim + j) = g.dk(i, j);
            dv(i, h * kHeadDim + j) = g.dv(i, j);
          }
      }
      Matrix dwq_shard = x.transpose().matmul(dq);
      Matrix dwk_shard = x.transpose().matmul(dk);
      Matrix dwv_shard = x.transpose().matmul(dv);
      Matrix dx_partial = dq.matmul(wq_shard.transpose()) + dk.matmul(wk_shard.transpose()) + dv.matmul(wv_shard.transpose());
      ring_allreduce(dx_partial.data(), dx_partial.size(), *ch);

      return RankResult{out, dwq_shard, dwk_shard, dwv_shard, dwout_shard, dx_partial};
    }));
  }

  std::vector<RankResult> rank_results;
  for (auto &f : results) rank_results.push_back(f.get());

  Matrix dwq_reassembled(kHidden, kHidden), dwk_reassembled(kHidden, kHidden), dwv_reassembled(kHidden, kHidden);
  Matrix dwout_reassembled(kHidden, kHidden);
  for (int r = 0; r < kWorldSize; ++r) {
    for (int i = 0; i < kHidden; ++i)
      for (int j = 0; j < kHiddenShard; ++j) {
        dwq_reassembled(i, r * kHiddenShard + j) = rank_results[static_cast<size_t>(r)].dwq_shard(i, j);
        dwk_reassembled(i, r * kHiddenShard + j) = rank_results[static_cast<size_t>(r)].dwk_shard(i, j);
        dwv_reassembled(i, r * kHiddenShard + j) = rank_results[static_cast<size_t>(r)].dwv_shard(i, j);
      }
    for (int i = 0; i < kHiddenShard; ++i)
      for (int j = 0; j < kHidden; ++j) dwout_reassembled(r * kHiddenShard + i, j) = rank_results[static_cast<size_t>(r)].dwout_shard(i, j);
  }

  bool ok = true;
  auto check = [&](const char *name, float diff, float tol = 1e-3f) {
    std::printf("  %-16s max abs diff = %.6f: %s\n", name, diff, diff <= tol ? "PASS" : "FAIL");
    if (diff > tol) ok = false;
  };
  check("forward out", max_abs_diff(ref.out, rank_results[0].out));
  check("dWq", max_abs_diff(ref.dwq, dwq_reassembled));
  check("dWk", max_abs_diff(ref.dwk, dwk_reassembled));
  check("dWv", max_abs_diff(ref.dwv, dwv_reassembled));
  check("dWout", max_abs_diff(ref.dwout, dwout_reassembled));
  check("dX", max_abs_diff(ref.dx, rank_results[0].dx));

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
