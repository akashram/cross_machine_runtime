// hyperband_test.cpp — real correctness checks: Successive Halving
// actually finds the best of several distinct deterministic-quality
// configs while consuming far less than the full training budget,
// Hyperband finds a near-best config among everything it samples across
// its brackets, and ASHA's asynchronous promotion rule actually only
// advances the top 1/eta fraction at each rung (verified against a
// hand-computable scenario) while still finding the true best config.
#include "hyperband.h"

#include <cmath>
#include <cstdio>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// Deterministic, resource-independent "quality" function: score is just
// config[0]. Lets tests know the exact right answer instead of relying
// on a real (slow, noisy) training run.
float deterministic_quality(const Config &c, int /*resource*/) { return c[0]; }

void test_successive_halving_finds_best_and_saves_resource() {
  std::vector<Config> configs = {{0.1f}, {0.9f}, {0.5f}, {0.3f}, {0.7f}, {0.2f}, {0.8f}, {0.4f}};
  SHAResult result = successive_halving(configs, 1, 8, 2.0f, deterministic_quality);

  int total_resource = 0;
  for (const auto &rec : result.history) total_resource += rec.resource;
  int full_training_cost = static_cast<int>(configs.size()) * 8;

  std::printf("  SHA best_config[0]=%.2f (true best=0.90), total_resource=%d (vs full-training cost=%d)\n",
              static_cast<double>(result.best_config[0]), total_resource, full_training_cost);
  require(std::fabs(result.best_config[0] - 0.9f) < 1e-6f, "Successive Halving finds the config with the true highest quality");
  require(total_resource < full_training_cost, "Successive Halving consumes less total resource than training every config to max_resource");
}

void test_hyperband_finds_near_best_among_sampled() {
  std::mt19937 seed_rng(7);
  ConfigSampler sampler = [](std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return Config{dist(rng)};
  };

  SHAResult result = hyperband(sampler, 27, 3.0f, deterministic_quality, 42);
  std::printf("  Hyperband best_config[0]=%.4f (true max is 1.0, but only ~%zu configs were ever sampled)\n",
              static_cast<double>(result.best_config[0]), result.history.size());
  require(result.best_score > 0.9f, "Hyperband finds a config with quality above 0.9 among everything it sampled across all brackets");
}

// A hand-computable ASHA scenario: 4 configs with distinct, known
// scores {0.1, 0.9, 0.5, 0.3}, min_resource=1, max_resource=4, eta=2.
// Rungs are 1, 2, 4. At rung 1 (all 4 configs), top-1/2 = 2 configs
// (0.9 and 0.5) should get promoted to rung 2; at rung 2 (those same 2
// configs), top-1/2 = 1 config (0.9) should get promoted to rung 4.
// The single truly-best config (0.9) must end up as the result, and
// total resource spent must be well under the 4*4=16 full-training cost
// since configs 0.1 and 0.3 never advance past rung 1.
void test_asha_promotes_only_top_fraction_and_finds_best() {
  std::vector<Config> fixed_configs = {{0.1f}, {0.9f}, {0.5f}, {0.3f}};
  std::size_t next = 0;
  ConfigSampler sampler = [&](std::mt19937 &) { return fixed_configs[next++]; };

  SHAResult result = asha(sampler, 1, 4, 2.0f, static_cast<int>(fixed_configs.size()), deterministic_quality, 1);

  int total_resource = 0;
  for (const auto &rec : result.history) total_resource += rec.resource;

  // Count how many distinct configs were ever evaluated at the top rung
  // (resource == 4) -- should be exactly 1 (only the true best).
  int at_top_rung = 0;
  for (const auto &rec : result.history)
    if (rec.resource == 4) ++at_top_rung;

  std::printf("  ASHA best_config[0]=%.2f, configs reaching top rung=%d, total_resource=%d (vs full-training cost=16)\n",
              static_cast<double>(result.best_config[0]), at_top_rung, total_resource);
  require(std::fabs(result.best_config[0] - 0.9f) < 1e-6f, "ASHA finds the single true-best config");
  require(at_top_rung == 1, "ASHA promotes exactly 1 of 4 configs all the way to the top rung (the top-1/2-of-top-1/2 = top 1/4)");
  require(total_resource < 16, "ASHA consumes less total resource than training every config to max_resource");
}

}  // namespace

int main() {
  test_successive_halving_finds_best_and_saves_resource();
  test_hyperband_finds_near_best_among_sampled();
  test_asha_promotes_only_top_fraction_and_finds_best();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
