// moe_test.cpp — 4 simulated ranks, each hosting one expert. Every rank
// routes its own tokens (top-1, shared gate weights) and calls
// moe_forward; the result must match a single-process reference that
// concatenates the global batch, routes it once, and applies each token's
// assigned expert directly — proof the dispatch/combine round-trip
// (variable-size all-to-all, twice) doesn't lose or misroute any token.
#include "moe_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <map>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

Matrix row_slice(const Matrix &m, int start, int count) {
  Matrix out(count, m.cols());
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}

float max_abs_diff(const Matrix &a, const Matrix &b) {
  float d = 0.0f;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j) d = std::max(d, std::abs(a(i, j) - b(i, j)));
  return d;
}

Matrix apply_expert(const Matrix &x, const Matrix &w, const Matrix &b) {
  return x.matmul(w).add_row_broadcast(b);
}

} // namespace

int main() {
  constexpr int kWorldSize = 4;
  constexpr int kHidden = 6;
  constexpr int kTokensPerRank = 5;
  constexpr int kTotalTokens = kWorldSize * kTokensPerRank; // 20
  constexpr uint16_t kBasePort = 35801;

  std::mt19937 rng(88);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };

  Matrix x_global = random_matrix(kTotalTokens, kHidden);
  Matrix w_gate = random_matrix(kHidden, kWorldSize); // shared/replicated router weights
  std::vector<Matrix> w_expert(kWorldSize), b_expert(kWorldSize);
  for (int e = 0; e < kWorldSize; ++e) {
    w_expert[static_cast<size_t>(e)] = random_matrix(kHidden, kHidden);
    b_expert[static_cast<size_t>(e)] = random_matrix(1, kHidden);
  }

  // --- Single-process reference: route the whole batch once, apply each
  // token's assigned expert directly.
  Matrix gate_logits_ref = x_global.matmul(w_gate);
  auto dest_ref = route_top1(gate_logits_ref);
  Matrix output_ref(kTotalTokens, kHidden);
  std::map<int, int> expert_load_ref;
  for (int i = 0; i < kTotalTokens; ++i) {
    int e = dest_ref[static_cast<size_t>(i)];
    expert_load_ref[e]++;
    Matrix row = row_slice(x_global, i, 1);
    Matrix out_row = apply_expert(row, w_expert[static_cast<size_t>(e)], b_expert[static_cast<size_t>(e)]);
    for (int j = 0; j < kHidden; ++j) output_ref(i, j) = out_row(0, j);
  }

  // --- Distributed: 4 ranks, each owning kTokensPerRank tokens and one expert.
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<Matrix>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> Matrix {
      Matrix x_local = row_slice(x_global, r * kTokensPerRank, kTokensPerRank);
      Matrix gate_logits = x_local.matmul(w_gate);
      auto dest = route_top1(gate_logits);

      auto expert_fn = [&](const Matrix &received) { return apply_expert(received, w_expert[static_cast<size_t>(r)], b_expert[static_cast<size_t>(r)]); };
      return moe_forward(x_local, dest, expert_fn, *ch);
    }));
  }

  Matrix output_distributed(kTotalTokens, kHidden);
  for (int r = 0; r < kWorldSize; ++r) {
    Matrix out = results[static_cast<size_t>(r)].get();
    for (int i = 0; i < kTokensPerRank; ++i)
      for (int j = 0; j < kHidden; ++j) output_distributed(r * kTokensPerRank + i, j) = out(i, j);
  }

  float diff = max_abs_diff(output_ref, output_distributed);
  bool ok = diff <= 1e-4f;
  std::printf("MoE dispatch/combine round-trip: max abs diff vs single-process reference = %.6f: %s\n", diff,
              ok ? "PASS" : "FAIL");

  std::printf("expert load (tokens routed to each expert, out of %d total):\n", kTotalTokens);
  for (int e = 0; e < kWorldSize; ++e) {
    std::printf("  expert %d: %d tokens (%.1f%%)\n", e, expert_load_ref[e],
                100.0 * expert_load_ref[e] / kTotalTokens);
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
