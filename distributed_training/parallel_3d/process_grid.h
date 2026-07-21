#pragma once

// 3D parallelism: PLAN.md step 15. Combines data (step 3), tensor (steps
// 11-13), and pipeline (step 14) parallelism, each already validated in
// isolation. The genuinely NEW thing this step introduces is the rank <->
// process-group math: with world_size = dp_size * tp_size * pp_size, every
// global rank belongs to exactly one DP group, one TP group, and one PP
// group simultaneously — getting the coordinate convention or group
// membership wrong silently cross-wires communicators (a rank
// all-reducing gradients with the wrong peers is not something that
// crashes, it just trains a subtly wrong model), which is exactly the
// kind of bug worth a dedicated, exhaustive combinatorial test rather than
// "it compiled."
//
// Convention (documented since it's a real design choice, not the only
// valid one): global_rank = dp_rank*(tp_size*pp_size) + tp_rank*pp_size +
// pp_rank — DP slowest-varying, PP fastest-varying. A TP group holds every
// rank sharing (dp, pp) and varying tp; a DP group holds every rank
// sharing (tp, pp) and varying dp; a PP group holds every rank sharing
// (dp, tp) and varying pp.

#include <vector>

namespace distributed_training {

struct ProcessGrid {
  int dp_size, tp_size, pp_size;

  int world_size() const { return dp_size * tp_size * pp_size; }

  int rank_of(int dp, int tp, int pp) const { return dp * (tp_size * pp_size) + tp * pp_size + pp; }

  struct Coord {
    int dp, tp, pp;
  };

  Coord coord_of(int global_rank) const {
    int pp = global_rank % pp_size;
    int rest = global_rank / pp_size;
    int tp = rest % tp_size;
    int dp = rest / tp_size;
    return Coord{dp, tp, pp};
  }

  std::vector<int> tp_group(int dp, int pp) const {
    std::vector<int> group;
    for (int tp = 0; tp < tp_size; ++tp) group.push_back(rank_of(dp, tp, pp));
    return group;
  }

  std::vector<int> dp_group(int tp, int pp) const {
    std::vector<int> group;
    for (int dp = 0; dp < dp_size; ++dp) group.push_back(rank_of(dp, tp, pp));
    return group;
  }

  std::vector<int> pp_group(int dp, int tp) const {
    std::vector<int> group;
    for (int pp = 0; pp < pp_size; ++pp) group.push_back(rank_of(dp, tp, pp));
    return group;
  }
};

} // namespace distributed_training
