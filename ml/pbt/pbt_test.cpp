// pbt_test.cpp — real correctness checks: the population's best score
// is non-decreasing round over round (the top-ranked member is never
// exploited FROM as a destination, so it always just keeps training),
// exploit actually clones the source member's state (verified via a toy
// "position" state vector), and -- PLAN.md step 18's central empirical
// claim -- PBT starting an entire population at a deliberately bad,
// slow-converging hyperparameter recovers via mutation and reaches a
// meaningfully better final score than holding that same bad
// hyperparameter fixed for the identical total training budget.
#include "pbt.h"

#include <cmath>
#include <cstdio>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// A toy "model": position moves toward target=10.0 by
// `position += step_size * (target - position)` each training step.
// step_size in (0, 2) converges (geometrically for (0,1), with
// decaying oscillation for (1,2)); this is a simple, analytically
// understood stand-in for "training" that a hyperparameter (step_size)
// can genuinely help or hurt, exactly the shape PBT is meant to tune.
constexpr float kTarget = 10.0f;

void test_population_best_is_non_decreasing() {
  std::vector<float> positions = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<PBTMember> population = {
      {{0.05f}, 0.0f}, {{0.3f}, 0.0f}, {{0.9f}, 0.0f}, {{1.5f}, 0.0f}};

  TrainStepFn train_step = [&](int idx, const std::vector<float> &hyperparams, int steps) {
    float step_size = hyperparams[0];
    for (int i = 0; i < steps; ++i) positions[static_cast<std::size_t>(idx)] += step_size * (kTarget - positions[static_cast<std::size_t>(idx)]);
    return -std::fabs(kTarget - positions[static_cast<std::size_t>(idx)]);
  };
  CloneStateFn clone_state = [&](int dst, int src) { positions[static_cast<std::size_t>(dst)] = positions[static_cast<std::size_t>(src)]; };

  PBTParams p;
  p.n_rounds = 10;
  p.steps_per_round = 3;
  p.random_state = 1;
  PBTResult result = population_based_training(population, train_step, clone_state, p);

  bool non_decreasing = true;
  for (std::size_t i = 1; i < result.best_score_per_round.size(); ++i)
    if (result.best_score_per_round[i] < result.best_score_per_round[i - 1] - 1e-5f) non_decreasing = false;

  std::printf("  best_score_per_round: ");
  for (float s : result.best_score_per_round) std::printf("%.3f ", static_cast<double>(s));
  std::printf("\n");
  require(non_decreasing, "the population's best score never decreases round over round (the top member is never overwritten by exploit)");
}

// A hand-verifiable exploit check: force member 0 (bad hyperparams,
// stuck near the start) and member 1 (good hyperparams, near target) by
// running a short warm-up, then run exactly one more PBT round with
// exploit fractions covering the whole population and confirm member
// 0's position actually became a copy of member 1's (state was really
// cloned, not just scored).
void test_exploit_actually_clones_state() {
  std::vector<float> positions = {0.0f, 0.0f};
  TrainStepFn train_step = [&](int idx, const std::vector<float> &hyperparams, int steps) {
    float step_size = hyperparams[0];
    for (int i = 0; i < steps; ++i) positions[static_cast<std::size_t>(idx)] += step_size * (kTarget - positions[static_cast<std::size_t>(idx)]);
    return -std::fabs(kTarget - positions[static_cast<std::size_t>(idx)]);
  };
  CloneStateFn clone_state = [&](int dst, int src) { positions[static_cast<std::size_t>(dst)] = positions[static_cast<std::size_t>(src)]; };

  std::vector<PBTMember> population = {{{0.001f}, 0.0f}, {{0.9f}, 0.0f}};
  PBTParams p;
  p.n_rounds = 2;  // round 0 trains + exploits; round 1 just trains (no more mutation) so we can inspect a stable post-exploit state
  p.steps_per_round = 5;
  p.exploit_bottom_fraction = 0.5f;
  p.exploit_top_fraction = 0.5f;
  p.random_state = 2;
  population_based_training(population, train_step, clone_state, p);

  std::printf("  after exploit+1 more training step: position[0]=%.4f, position[1]=%.4f\n", static_cast<double>(positions[0]),
              static_cast<double>(positions[1]));
  require(std::fabs(positions[0] - positions[1]) < 3.0f,
          "the slow member's position ends up close to the fast member's after exploit clones its state (not still stuck near 0)");
}

// PLAN.md's central claim: starting an ENTIRE population at a
// deliberately bad, slow-converging step_size, PBT's mutation should
// discover better step sizes and end up meaningfully closer to target
// than holding that same bad step_size fixed for the identical total
// training budget (n_rounds * steps_per_round steps per member either
// way).
void test_pbt_beats_fixed_bad_hyperparameter() {
  const float bad_step = 0.01f;
  const int n_members = 6;
  const int n_rounds = 12;
  const int steps_per_round = 5;

  // Fixed baseline: no PBT, every member trains alone with the same bad
  // step_size for the full budget.
  float fixed_final_position = 0.0f;
  for (int i = 0; i < n_rounds * steps_per_round; ++i) fixed_final_position += bad_step * (kTarget - fixed_final_position);
  float fixed_score = -std::fabs(kTarget - fixed_final_position);

  // PBT: population of n_members, all starting at the same bad step_size.
  std::vector<float> positions(n_members, 0.0f);
  TrainStepFn train_step = [&](int idx, const std::vector<float> &hyperparams, int steps) {
    float step_size = hyperparams[0];
    for (int i = 0; i < steps; ++i) positions[static_cast<std::size_t>(idx)] += step_size * (kTarget - positions[static_cast<std::size_t>(idx)]);
    return -std::fabs(kTarget - positions[static_cast<std::size_t>(idx)]);
  };
  CloneStateFn clone_state = [&](int dst, int src) { positions[static_cast<std::size_t>(dst)] = positions[static_cast<std::size_t>(src)]; };

  std::vector<PBTMember> population(static_cast<std::size_t>(n_members), PBTMember{{bad_step}, 0.0f});
  PBTParams p;
  p.n_rounds = n_rounds;
  p.steps_per_round = steps_per_round;
  p.random_state = 5;
  PBTResult result = population_based_training(population, train_step, clone_state, p);

  std::printf("  fixed-bad-step_size final score=%.4f, PBT final score=%.4f (both %d total training steps per member)\n",
              static_cast<double>(fixed_score), static_cast<double>(result.best_score_per_round.back()), n_rounds * steps_per_round);
  require(result.best_score_per_round.back() > fixed_score,
          "PBT (mutating step_size via exploit/explore) reaches a better final score than holding the same bad step_size fixed");
}

}  // namespace

int main() {
  test_population_best_is_non_decreasing();
  test_exploit_actually_clones_state();
  test_pbt_beats_fixed_bad_hyperparameter();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
