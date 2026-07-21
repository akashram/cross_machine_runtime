// parallel_3d_test.cpp — two things:
//  1. Exhaustive combinatorial validation of ProcessGrid: rank<->coordinate
//     is a bijection, and every group (TP/DP/PP) partitions the full rank
//     set correctly (right size, right membership, no overlaps, no gaps).
//  2. A composed end-to-end demo: DP(2) x TP(4) (PP fixed at 1 — pipeline
//     parallelism stays validated at the scheduling level from step 14,
//     not executed here; see README for why composing real pipeline
//     STAGE EXECUTION was out of scope for this step). Reuses step 11's
//     tensor-parallel linear within each TP group and step 3's
//     ring_allreduce pattern across each DP group, verifying the DP
//     all-reduce only combines gradients within the correct TP shard —
//     not across shards, which would silently corrupt the model.
#include "process_grid.h"
#include "../col_row_linear/tensor_parallel_linear.h"

#include "ring_allreduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <set>
#include <vector>

using namespace distributed_training;

namespace {

bool test_process_grid() {
  bool ok = true;
  struct Config { int dp, tp, pp; };
  std::vector<Config> configs{{2, 4, 1}, {2, 2, 2}, {3, 1, 2}, {1, 1, 1}, {4, 3, 2}};

  for (auto &cfg : configs) {
    ProcessGrid grid{cfg.dp, cfg.tp, cfg.pp};
    int n = grid.world_size();

    // Bijection: rank_of <-> coord_of round-trips for every rank.
    std::set<int> seen_ranks;
    for (int dp = 0; dp < cfg.dp; ++dp) {
      for (int tp = 0; tp < cfg.tp; ++tp) {
        for (int pp = 0; pp < cfg.pp; ++pp) {
          int r = grid.rank_of(dp, tp, pp);
          if (r < 0 || r >= n) { ok = false; continue; }
          if (!seen_ranks.insert(r).second) ok = false; // collision
          auto c = grid.coord_of(r);
          if (c.dp != dp || c.tp != tp || c.pp != pp) ok = false;
        }
      }
    }
    if (static_cast<int>(seen_ranks.size()) != n) ok = false; // covers every rank exactly once

    // Every TP group: right size, members share (dp,pp), distinct tp values covering [0,tp).
    for (int dp = 0; dp < cfg.dp; ++dp) {
      for (int pp = 0; pp < cfg.pp; ++pp) {
        auto group = grid.tp_group(dp, pp);
        if (static_cast<int>(group.size()) != cfg.tp) ok = false;
        std::set<int> tp_vals;
        for (int r : group) {
          auto c = grid.coord_of(r);
          if (c.dp != dp || c.pp != pp) ok = false;
          tp_vals.insert(c.tp);
        }
        if (static_cast<int>(tp_vals.size()) != cfg.tp) ok = false;
      }
    }
    // Same shape checks for DP and PP groups.
    for (int tp = 0; tp < cfg.tp; ++tp) {
      for (int pp = 0; pp < cfg.pp; ++pp) {
        auto group = grid.dp_group(tp, pp);
        if (static_cast<int>(group.size()) != cfg.dp) ok = false;
        std::set<int> dp_vals;
        for (int r : group) {
          auto c = grid.coord_of(r);
          if (c.tp != tp || c.pp != pp) ok = false;
          dp_vals.insert(c.dp);
        }
        if (static_cast<int>(dp_vals.size()) != cfg.dp) ok = false;
      }
    }
    for (int dp = 0; dp < cfg.dp; ++dp) {
      for (int tp = 0; tp < cfg.tp; ++tp) {
        auto group = grid.pp_group(dp, tp);
        if (static_cast<int>(group.size()) != cfg.pp) ok = false;
        std::set<int> pp_vals;
        for (int r : group) {
          auto c = grid.coord_of(r);
          if (c.dp != dp || c.tp != tp) ok = false;
          pp_vals.insert(c.pp);
        }
        if (static_cast<int>(pp_vals.size()) != cfg.pp) ok = false;
      }
    }

    std::printf("  grid dp=%d tp=%d pp=%d (world_size=%d): %s\n", cfg.dp, cfg.tp, cfg.pp, n, ok ? "PASS" : "FAIL");
  }
  return ok;
}

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

bool test_composed_dp_tp() {
  constexpr int kDp = 2, kTp = 4, kPp = 1;
  constexpr int kIn = 8, kHidden = 16, kOut = 8;
  constexpr int kHiddenShard = kHidden / kTp;
  ProcessGrid grid{kDp, kTp, kPp};

  // Two separate 4-rank TP meshes (one per DP replica) and four separate
  // 2-rank DP meshes (one per TP shard, since pp_size==1) -- mirroring how
  // real multi-dimensional parallelism uses a SEPARATE communicator per
  // dimension, not one communicator overloaded for everything.
  std::vector<std::vector<std::unique_ptr<netcommon::Channel>>> tp_meshes(static_cast<size_t>(kDp));
  for (int dp = 0; dp < kDp; ++dp) tp_meshes[static_cast<size_t>(dp)] = netcommon::make_tcp_loopback_mesh(kTp, static_cast<uint16_t>(35601 + dp * 10));
  std::vector<std::vector<std::unique_ptr<netcommon::Channel>>> dp_meshes(static_cast<size_t>(kTp));
  for (int tp = 0; tp < kTp; ++tp) dp_meshes[static_cast<size_t>(tp)] = netcommon::make_tcp_loopback_mesh(kDp, static_cast<uint16_t>(35701 + tp * 10));

  std::mt19937 rng(77);
  std::normal_distribution<float> dist(0.0f, 0.3f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };
  Matrix w_col = random_matrix(kIn, kHidden), b_col(1, kHidden), w_row = random_matrix(kHidden, kOut);
  for (int j = 0; j < kHidden; ++j) b_col(0, j) = dist(rng);

  // Two DP replicas see DIFFERENT data (the whole point of data
  // parallelism), sharded identically by tensor parallelism.
  Matrix x_replica0 = random_matrix(6, kIn), x_replica1 = random_matrix(6, kIn);

  std::vector<std::future<Matrix>> results(static_cast<size_t>(grid.world_size()));
  for (int dp = 0; dp < kDp; ++dp) {
    for (int tp = 0; tp < kTp; ++tp) {
      int r = grid.rank_of(dp, tp, 0);
      netcommon::Channel *tp_ch = tp_meshes[static_cast<size_t>(dp)][static_cast<size_t>(tp)].get();
      netcommon::Channel *dp_ch = dp_meshes[static_cast<size_t>(tp)][static_cast<size_t>(dp)].get();
      const Matrix &x = (dp == 0) ? x_replica0 : x_replica1;
      results[static_cast<size_t>(r)] = std::async(std::launch::async, [&, tp, tp_ch, dp_ch]() -> Matrix {
        ColumnParallelLinear col{col_slice(w_col, tp * kHiddenShard, kHiddenShard), col_slice(b_col, tp * kHiddenShard, kHiddenShard)};
        RowParallelLinear row{row_slice(w_row, tp * kHiddenShard, kHiddenShard)};

        Matrix pre = col.forward(x);
        Matrix act = pre.apply([](float v) { return v > 0.0f ? v : 0.0f; });
        Matrix partial = row.forward(act);
        ring_allreduce(partial.data(), partial.size(), *tp_ch); // TP all-reduce: within this DP replica's TP group only
        Matrix out = partial;

        // Toy gradient: dOut = out (loss = 0.5*sum(out^2)), backprop only
        // to this rank's W_col shard (enough to demonstrate DP all-reduce
        // stays within the correct TP shard).
        auto row_grads = row.backward(act, out);
        Matrix d_act = row_grads.dx_shard;
        Matrix d_pre = d_act.elementwise_mul(pre.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; }));
        auto col_grads = col.backward(x, d_pre);
        Matrix dw_col_shard = col_grads.dweight;

        // DP all-reduce: average this TP shard's gradient across the 2 DP
        // replicas that own the SAME shard (tp fixed, dp varies) -- using
        // the per-tp DP mesh, never the TP mesh.
        ring_allreduce(dw_col_shard.data(), dw_col_shard.size(), *dp_ch);
        dw_col_shard = dw_col_shard * 0.5f; // sum -> mean over 2 replicas
        return dw_col_shard;
      });
    }
  }

  // Reference: for a given replica's x, compute the FULL (all-shard)
  // forward first — exactly like tensor_parallel_linear_test.cpp's
  // reference — to get the true dOut (the TP-all-reduced output), THEN
  // backward through one specific shard using that true dOut. This must
  // match what the distributed code above actually does (dOut = out,
  // the post-all-reduce value), not each shard's own pre-all-reduce
  // partial sum, which is a different (and wrong) quantity.
  auto local_grad_for_shard = [&](const Matrix &x, int target_tp) {
    Matrix out_full(x.rows(), kOut);
    Matrix target_act, target_pre;
    for (int tp = 0; tp < kTp; ++tp) {
      Matrix w_col_shard = col_slice(w_col, tp * kHiddenShard, kHiddenShard);
      Matrix b_col_shard = col_slice(b_col, tp * kHiddenShard, kHiddenShard);
      Matrix w_row_shard = row_slice(w_row, tp * kHiddenShard, kHiddenShard);
      ColumnParallelLinear col{w_col_shard, b_col_shard};
      RowParallelLinear row{w_row_shard};
      Matrix pre = col.forward(x);
      Matrix act = pre.apply([](float v) { return v > 0.0f ? v : 0.0f; });
      out_full = out_full + row.forward(act);
      if (tp == target_tp) { target_act = act; target_pre = pre; }
    }
    Matrix d_out = out_full; // loss = 0.5*sum(out^2)
    RowParallelLinear target_row{row_slice(w_row, target_tp * kHiddenShard, kHiddenShard)};
    ColumnParallelLinear target_col{col_slice(w_col, target_tp * kHiddenShard, kHiddenShard),
                                     col_slice(b_col, target_tp * kHiddenShard, kHiddenShard)};
    auto row_grads = target_row.backward(target_act, d_out);
    Matrix d_pre = row_grads.dx_shard.elementwise_mul(target_pre.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; }));
    return target_col.backward(x, d_pre).dweight;
  };

  bool ok = true;
  for (int tp = 0; tp < kTp; ++tp) {
    Matrix expected = (local_grad_for_shard(x_replica0, tp) + local_grad_for_shard(x_replica1, tp)) * 0.5f;
    int r0 = grid.rank_of(0, tp, 0);
    Matrix actual = results[static_cast<size_t>(r0)].get();
    float max_diff = 0.0f;
    for (int i = 0; i < expected.rows(); ++i)
      for (int j = 0; j < expected.cols(); ++j) max_diff = std::max(max_diff, std::abs(expected(i, j) - actual(i, j)));
    bool shard_ok = max_diff <= 1e-3f;
    std::printf("  tp shard %d: DP-averaged dW_col max abs diff vs reference = %.6f: %s\n", tp, max_diff,
                shard_ok ? "PASS" : "FAIL");
    ok = ok && shard_ok;
  }
  // Drain the other DP replica's futures too (avoid std::future destructor blocking on an unread result being an issue elsewhere).
  for (int tp = 0; tp < kTp; ++tp) results[static_cast<size_t>(grid.rank_of(1, tp, 0))].get();

  return ok;
}

} // namespace

int main() {
  std::printf("test 1 (process grid: rank<->coordinate bijection, group partitioning):\n");
  bool ok = test_process_grid();

  std::printf("\ntest 2 (composed DP(2) x TP(4), PP fixed at 1 -- see README):\n");
  ok = test_composed_dp_tp() && ok;

  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
