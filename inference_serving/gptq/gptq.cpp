#include "gptq.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace inference_serving {

namespace {

// [min_val, max_val] -> per-group affine quantization params: scale =
// (max-min)/(2^bits-1), zero = round(-min/scale), clamped into the valid
// int range. w ≈ (q - zero) * scale, q in [0, 2^bits - 1].
struct QuantParams {
  float scale;
  int32_t zero;
};

QuantParams find_group_params(float min_val, float max_val, int bits) {
  int qmax = (1 << bits) - 1;
  // A degenerate all-equal group would divide by zero; fall back to a
  // scale of 1 (any nonzero value avoids NaN, and the group's quantized
  // values will all collapse to the same code anyway, exactly correct
  // for a constant group).
  float range = max_val - min_val;
  float scale = (range > 1e-8f) ? (range / static_cast<float>(qmax)) : 1.0f;
  int32_t zero = static_cast<int32_t>(std::lround(-min_val / scale));
  zero = std::clamp(zero, 0, qmax);
  return {scale, zero};
}

int32_t quantize_scalar(float w, const QuantParams &p, int bits) {
  int qmax = (1 << bits) - 1;
  int32_t q = static_cast<int32_t>(std::lround(w / p.scale)) + p.zero;
  return std::clamp(q, 0, qmax);
}

float dequantize_scalar(int32_t q, const QuantParams &p) { return (static_cast<float>(q) - p.zero) * p.scale; }

// H = 2 * X^T X + damp * mean(diag(X^T X)) * I  — the calibration
// Hessian for a layer whose input activations are X [samples x cols].
// Dampening avoids a singular/ill-conditioned H when samples < cols or
// activations are collinear (both true for this repo's tiny calibration
// sets), the same practical fix the GPTQ reference implementation uses.
std::vector<std::vector<double>> compute_hessian(const Matrix &activations, double damp_frac) {
  int cols = activations.cols();
  int samples = activations.rows();
  std::vector<std::vector<double>> h(static_cast<std::size_t>(cols), std::vector<double>(static_cast<std::size_t>(cols), 0.0));
  for (int s = 0; s < samples; ++s) {
    for (int i = 0; i < cols; ++i) {
      double xi = static_cast<double>(activations(s, i));
      if (xi == 0.0) continue;
      for (int j = 0; j < cols; ++j)
        h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] += 2.0 * xi * static_cast<double>(activations(s, j));
    }
  }
  double diag_mean = 0.0;
  for (int i = 0; i < cols; ++i) diag_mean += h[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
  diag_mean = (cols > 0) ? diag_mean / cols : 0.0;
  double damp = damp_frac * diag_mean;
  for (int i = 0; i < cols; ++i) h[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] += damp;
  return h;
}

// Gauss-Jordan inversion, double precision. Calibration Hessians here are
// small (cols == transformer d_model, tens not thousands) so O(cols^3)
// direct inversion is fine — the real GPTQ implementation uses a Cholesky
// factorization for speed at LLM scale, an optimization orthogonal to
// the algorithm this step demonstrates.
std::vector<std::vector<double>> invert(std::vector<std::vector<double>> a) {
  int n = static_cast<int>(a.size());
  std::vector<std::vector<double>> inv(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) inv[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(a[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]);
    for (int r = col + 1; r < n; ++r) {
      double v = std::abs(a[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)]);
      if (v > best) { best = v; pivot = r; }
    }
    std::swap(a[static_cast<std::size_t>(col)], a[static_cast<std::size_t>(pivot)]);
    std::swap(inv[static_cast<std::size_t>(col)], inv[static_cast<std::size_t>(pivot)]);

    double diag = a[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
    if (std::abs(diag) < 1e-12) diag = (diag >= 0 ? 1e-12 : -1e-12);  // dampening above should prevent this in practice
    for (int j = 0; j < n; ++j) {
      a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= diag;
      inv[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= diag;
    }
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      double factor = a[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)];
      if (factor == 0.0) continue;
      for (int j = 0; j < n; ++j) {
        a[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] -= factor * a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
        inv[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] -= factor * inv[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
      }
    }
  }
  return inv;
}

}  // namespace

QuantizedWeight quantize_rtn(const Matrix &weight, int group_size, int bits) {
  QuantizedWeight q;
  q.rows = weight.rows();
  q.cols = weight.cols();
  q.group_size = group_size;
  q.bits = bits;
  q.qweight.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(q.cols), 0);
  int groups = q.num_groups();
  q.scales.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(groups), 0.0f);
  q.zeros.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(groups), 0);

  for (int r = 0; r < q.rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      int c0 = g * group_size, c1 = std::min(c0 + group_size, q.cols);
      float min_v = std::numeric_limits<float>::max(), max_v = std::numeric_limits<float>::lowest();
      for (int c = c0; c < c1; ++c) { min_v = std::min(min_v, weight(r, c)); max_v = std::max(max_v, weight(r, c)); }
      QuantParams p = find_group_params(min_v, max_v, bits);
      q.scales[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)] = p.scale;
      q.zeros[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)] = p.zero;
      for (int c = c0; c < c1; ++c)
        q.qweight[static_cast<std::size_t>(r) * static_cast<std::size_t>(q.cols) + static_cast<std::size_t>(c)] =
            quantize_scalar(weight(r, c), p, bits);
    }
  }
  return q;
}

Matrix dequantize(const QuantizedWeight &q) {
  Matrix out(q.rows, q.cols);
  int groups = q.num_groups();
  for (int r = 0; r < q.rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      int c0 = g * q.group_size, c1 = std::min(c0 + q.group_size, q.cols);
      float scale = q.scales[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)];
      int32_t zero = q.zeros[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)];
      QuantParams p{scale, zero};
      for (int c = c0; c < c1; ++c)
        out(r, c) = dequantize_scalar(q.qweight[static_cast<std::size_t>(r) * static_cast<std::size_t>(q.cols) + static_cast<std::size_t>(c)], p);
    }
  }
  return out;
}

QuantizedWeight GptqQuantizer::quantize(const Matrix &weight, const Matrix &calibration_activations) const {
  // weight must be [out_features x in_features] (standard nn.Linear
  // layout) with in_features == calibration_activations' column count --
  // the Hessian is computed over the layer's INPUT dimension. Passing a
  // weight in the opposite ([in x out]) layout is a caller bug (see
  // gptq_test.cpp's transformer/ w_out transpose for a real example of
  // getting this right); assert instead of silently indexing out of
  // bounds into a differently-shaped Hessian.
  assert(weight.cols() == calibration_activations.cols() &&
         "GptqQuantizer::quantize: weight.cols() must equal calibration_activations.cols() (both = in_features)");
  QuantizedWeight q;
  q.rows = weight.rows();
  q.cols = weight.cols();
  q.group_size = group_size_;
  q.bits = bits_;
  int groups = q.num_groups();
  q.qweight.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(q.cols), 0);
  q.scales.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(groups), 0.0f);
  q.zeros.assign(static_cast<std::size_t>(q.rows) * static_cast<std::size_t>(groups), 0);

  auto h = compute_hessian(calibration_activations, /*damp_frac=*/0.01);
  auto hinv = invert(h);

  for (int r = 0; r < q.rows; ++r) {
    // Working copy of this row -- error compensation mutates it in place
    // as each column is quantized (columns to the right of the current
    // one absorb the current column's rounding error).
    std::vector<double> w(static_cast<std::size_t>(q.cols));
    for (int c = 0; c < q.cols; ++c) w[static_cast<std::size_t>(c)] = static_cast<double>(weight(r, c));

    for (int c = 0; c < q.cols; ++c) {
      int g = c / group_size_;
      // A new group's scale/zero is computed from this group's CURRENT
      // values (which may already reflect compensation pushed in from
      // an earlier group's quantization) -- matches the reference GPTQ
      // implementation's find_params-at-group-start behavior.
      if (c % group_size_ == 0) {
        int c0 = c, c1 = std::min(c0 + group_size_, q.cols);
        float min_v = std::numeric_limits<float>::max(), max_v = std::numeric_limits<float>::lowest();
        for (int cc = c0; cc < c1; ++cc) {
          min_v = std::min(min_v, static_cast<float>(w[static_cast<std::size_t>(cc)]));
          max_v = std::max(max_v, static_cast<float>(w[static_cast<std::size_t>(cc)]));
        }
        QuantParams p = find_group_params(min_v, max_v, bits_);
        q.scales[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)] = p.scale;
        q.zeros[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)] = p.zero;
      }
      float scale = q.scales[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)];
      int32_t zero = q.zeros[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)];
      QuantParams p{scale, zero};

      int32_t qval = quantize_scalar(static_cast<float>(w[static_cast<std::size_t>(c)]), p, bits_);
      q.qweight[static_cast<std::size_t>(r) * static_cast<std::size_t>(q.cols) + static_cast<std::size_t>(c)] = qval;

      double recon = static_cast<double>(dequantize_scalar(qval, p));
      double err = (w[static_cast<std::size_t>(c)] - recon) / hinv[static_cast<std::size_t>(c)][static_cast<std::size_t>(c)];
      // Push this column's rounding error onto every not-yet-quantized
      // column, weighted by the inverse Hessian's off-diagonal -- the
      // step that distinguishes GPTQ from plain round-to-nearest.
      for (int j = c + 1; j < q.cols; ++j)
        w[static_cast<std::size_t>(j)] -= err * hinv[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)];
    }
  }
  return q;
}

}  // namespace inference_serving
