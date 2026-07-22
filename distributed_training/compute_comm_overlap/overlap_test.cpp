// overlap_test.cpp — 4 simulated data-parallel ranks, each backpropping
// through a 4-layer linear chain (real, sequentially-dependent per-layer
// gradient computation: layer i's gradient needs layer i+1's delta).
// overlapped_backward's result must exactly match serial_backward's for
// every rank — overlap changes WHEN communication happens, never the
// result. Wall-clock timing is reported as information only (see
// checkpoint/README.md for why asserting a speedup on this dev machine
// is fighting the test environment, not validating logic — the same
// reasoning applies here).
#include "overlap_backward.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

constexpr int kNumLayers = 4;
constexpr int kHidden = 8;
constexpr int kBatch = 6;

// Builds a FRESH forward pass + backward closure over independent, owned
// state each call — so overlapped_backward and serial_backward each get
// their own untouched copy, even though both are exercising the "same"
// model and data.
std::function<Matrix(int)> make_compute_fn(const Matrix &x, const std::vector<Matrix> &weights) {
  auto activations = std::make_shared<std::vector<Matrix>>(static_cast<size_t>(kNumLayers) + 1);
  (*activations)[0] = x;
  for (int i = 0; i < kNumLayers; ++i) (*activations)[static_cast<size_t>(i + 1)] = (*activations)[static_cast<size_t>(i)].matmul(weights[static_cast<size_t>(i)]);

  auto delta = std::make_shared<Matrix>((*activations)[static_cast<size_t>(kNumLayers)]); // dLoss/da_L, loss = 0.5*sum(a_L^2)
  auto acts = activations;
  auto w = std::make_shared<std::vector<Matrix>>(weights);

  return [acts, delta, w](int layer) -> Matrix {
    Matrix dW = (*acts)[static_cast<size_t>(layer)].transpose().matmul(*delta);
    if (layer > 0) *delta = delta->matmul((*w)[static_cast<size_t>(layer)].transpose());
    return dW;
  };
}

float max_abs_diff(const Matrix &a, const Matrix &b) {
  float d = 0.0f;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j) d = std::max(d, std::abs(a(i, j) - b(i, j)));
  return d;
}

} // namespace

int main() {
  constexpr int kWorldSize = 4;
  constexpr uint16_t kBasePortOverlap = 35901;
  constexpr uint16_t kBasePortSerial = 35951;

  std::mt19937 rng(66);
  std::normal_distribution<float> dist(0.0f, 0.3f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };

  std::vector<Matrix> weights(kNumLayers);
  for (auto &w : weights) w = random_matrix(kHidden, kHidden);

  std::vector<Matrix> x_per_rank(kWorldSize);
  for (auto &x : x_per_rank) x = random_matrix(kBatch, kHidden);

  auto overlap_channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePortOverlap);
  auto serial_channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePortSerial);

  std::vector<std::future<std::pair<std::vector<Matrix>, std::vector<Matrix>>>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *overlap_ch = overlap_channels[static_cast<size_t>(r)].get();
    netcommon::Channel *serial_ch = serial_channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, overlap_ch, serial_ch]() {
      auto compute_fn_overlap = make_compute_fn(x_per_rank[static_cast<size_t>(r)], weights);
      auto overlap_result = overlapped_backward(kNumLayers, compute_fn_overlap, *overlap_ch);

      auto compute_fn_serial = make_compute_fn(x_per_rank[static_cast<size_t>(r)], weights);
      auto serial_result = serial_backward(kNumLayers, compute_fn_serial, *serial_ch);

      return std::make_pair(overlap_result, serial_result);
    }));
  }

  bool ok = true;
  for (int r = 0; r < kWorldSize; ++r) {
    auto [overlap_result, serial_result] = results[static_cast<size_t>(r)].get();
    for (int layer = 0; layer < kNumLayers; ++layer) {
      float diff = max_abs_diff(overlap_result[static_cast<size_t>(layer)], serial_result[static_cast<size_t>(layer)]);
      if (diff > 1e-4f) ok = false;
      if (r == 0) std::printf("  rank 0, layer %d: overlapped vs serial max abs diff = %.6f\n", layer, diff);
    }
  }
  std::printf("test 1 (overlap produces identical results to serial, all ranks): %s\n", ok ? "PASS" : "FAIL");

  // Informational timing only (see file comment for why not asserted).
  auto timed_channels_a = netcommon::make_tcp_loopback_mesh(kWorldSize, static_cast<uint16_t>(36001));
  auto timed_channels_b = netcommon::make_tcp_loopback_mesh(kWorldSize, static_cast<uint16_t>(36051));
  std::vector<std::future<double>> overlap_times, serial_times;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch_a = timed_channels_a[static_cast<size_t>(r)].get();
    overlap_times.push_back(std::async(std::launch::async, [&, r, ch_a]() {
      auto fn = make_compute_fn(x_per_rank[static_cast<size_t>(r)], weights);
      auto start = std::chrono::steady_clock::now();
      overlapped_backward(kNumLayers, fn, *ch_a);
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }));
  }
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch_b = timed_channels_b[static_cast<size_t>(r)].get();
    serial_times.push_back(std::async(std::launch::async, [&, r, ch_b]() {
      auto fn = make_compute_fn(x_per_rank[static_cast<size_t>(r)], weights);
      auto start = std::chrono::steady_clock::now();
      serial_backward(kNumLayers, fn, *ch_b);
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }));
  }
  double max_overlap_t = 0, max_serial_t = 0;
  for (auto &f : overlap_times) max_overlap_t = std::max(max_overlap_t, f.get());
  for (auto &f : serial_times) max_serial_t = std::max(max_serial_t, f.get());
  std::printf("informational timing (not asserted -- see file comment): overlapped=%.4fs serial=%.4fs\n", max_overlap_t,
              max_serial_t);

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
