//===- crossbar_mac.h - resistive crossbar analog matrix-vector multiply ===//
//
// PLAN.md Phase 17 step 2: simulate a resistive crossbar performing analog
// matrix-vector multiply via Ohm's law (current = conductance * voltage)
// and Kirchhoff's current law (currents into a row sum for free, on the
// physical wire -- no separate "add" step), with step 1's device
// non-idealities (`analog::ConductanceCell`) injected into every cell.
//
// Signed weights need a differential pair per matrix entry, since a
// physical conductance can't be negative: w_ij is encoded as a PAIR of
// cells (G_pos, G_neg), one of which is programmed to |w_ij|'s quantized
// level and the other left at the baseline (level 0), so that
// (G_pos - G_neg) = sign(w_ij) * |w_ij| * (g_max - g_min) exactly in the
// noiseless, infinite-precision limit -- this is the standard technique
// real crossbar accelerators use (ISAAC, PRIME; see READING_LIST.md's
// Phase 17 step 2 entry), not a simplification unique to this repo. Input
// voltages, unlike conductances, CAN be negative electrically, so the
// input vector needs no such encoding.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../device_model/device_model.h"

#include <cmath>
#include <vector>

namespace analog {

using Matrix = std::vector<std::vector<double>>; // row-major, weights in [-1, 1]

// Quantize |w| in [0,1] to a level index in [0, num_levels-1].
inline int quantize_level(double abs_w, int num_levels) {
  int level = static_cast<int>(std::round(abs_w * (num_levels - 1)));
  return std::clamp(level, 0, num_levels - 1);
}

class Crossbar {
public:
  Crossbar(int rows, int cols, DeviceParams params, uint64_t seed = 0)
      : rows_(rows), cols_(cols), params_(params) {
    pos_.reserve(static_cast<size_t>(rows));
    neg_.reserve(static_cast<size_t>(rows));
    uint64_t cell_seed = seed;
    for (int r = 0; r < rows; ++r) {
      std::vector<ConductanceCell> pos_row, neg_row;
      pos_row.reserve(static_cast<size_t>(cols));
      neg_row.reserve(static_cast<size_t>(cols));
      for (int c = 0; c < cols; ++c) {
        pos_row.emplace_back(params_, cell_seed++);
        neg_row.emplace_back(params_, cell_seed++);
      }
      pos_.push_back(std::move(pos_row));
      neg_.push_back(std::move(neg_row));
    }
  }

  // Program every cell to represent W (rows x cols, entries expected in
  // [-1, 1] -- values outside that range still work but saturate at the
  // crossbar's max representable magnitude).
  void program(const Matrix &W) {
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        double w = std::clamp(W[static_cast<size_t>(r)][static_cast<size_t>(c)], -1.0, 1.0);
        int level = quantize_level(std::abs(w), params_.num_levels);
        if (w >= 0.0) {
          pos_[static_cast<size_t>(r)][static_cast<size_t>(c)].write(level);
          neg_[static_cast<size_t>(r)][static_cast<size_t>(c)].write(0);
        } else {
          pos_[static_cast<size_t>(r)][static_cast<size_t>(c)].write(0);
          neg_[static_cast<size_t>(r)][static_cast<size_t>(c)].write(level);
        }
      }
    }
  }

  // Analog matrix-vector multiply: apply x as column voltages, read the
  // per-row summed current (KCL: the sum happens for free on the wire --
  // this loop is doing the READING and bookkeeping, not an "addition" the
  // real hardware would separately perform), decode back to weight scale.
  std::vector<double> multiply(const std::vector<double> &x, double elapsed = 0.0) const {
    double range = params_.g_max - params_.g_min;
    std::vector<double> y(static_cast<size_t>(rows_), 0.0);
    for (int r = 0; r < rows_; ++r) {
      double sum = 0.0;
      for (int c = 0; c < cols_; ++c) {
        double g_pos = pos_[static_cast<size_t>(r)][static_cast<size_t>(c)].read(elapsed);
        double g_neg = neg_[static_cast<size_t>(r)][static_cast<size_t>(c)].read(elapsed);
        sum += ((g_pos - g_neg) / range) * x[static_cast<size_t>(c)];
      }
      y[static_cast<size_t>(r)] = sum;
    }
    return y;
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

private:
  int rows_, cols_;
  DeviceParams params_;
  std::vector<std::vector<ConductanceCell>> pos_, neg_;
};

// Exact digital reference: y = W * x, no quantization or noise.
inline std::vector<double> ideal_matvec(const Matrix &W, const std::vector<double> &x) {
  size_t rows = W.size(), cols = x.size();
  std::vector<double> y(rows, 0.0);
  for (size_t r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (size_t c = 0; c < cols; ++c) sum += W[r][c] * x[c];
    y[r] = sum;
  }
  return y;
}

struct AccuracyResult {
  double rmse;
  double relative_rmse; // rmse / RMS(y_ideal)
};

inline AccuracyResult compare(const std::vector<double> &y_analog, const std::vector<double> &y_ideal) {
  double sq_err = 0.0, sq_ideal = 0.0;
  for (size_t i = 0; i < y_ideal.size(); ++i) {
    double d = y_analog[i] - y_ideal[i];
    sq_err += d * d;
    sq_ideal += y_ideal[i] * y_ideal[i];
  }
  double rmse = std::sqrt(sq_err / static_cast<double>(y_ideal.size()));
  double rms_ideal = std::sqrt(sq_ideal / static_cast<double>(y_ideal.size()));
  return {rmse, rmse / std::max(1e-12, rms_ideal)};
}

} // namespace analog
