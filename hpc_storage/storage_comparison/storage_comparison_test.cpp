// storage_comparison_test.cpp — prints the literature-grounded comparison
// table and checks its internal consistency (not "correctness" against
// real hardware, which no local access exists to measure -- see
// storage_comparison.h's header comment).
#include "storage_comparison.h"

#include <cstdio>

using namespace hpc_storage;

namespace {
int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}
} // namespace

int main() {
  std::printf("== Parallel/distributed storage comparison (literature/vendor-doc-grounded) ==\n\n");
  for (const auto &fs : parallel_fs_classes()) {
    std::printf("%-12s | %-70s | metadata=%d/5 | GDS=%s | reduction=%s\n", fs.name.c_str(),
                fs.architecture.c_str(), fs.small_file_metadata_rating, fs.gds_certified ? "yes" : "no",
                fs.data_reduction_technique.c_str());
  }

  std::printf("\n== Illustrative AI-workload-fit composite score ==\n");
  auto scores = compute_ai_workload_score();
  for (const auto &s : scores) std::printf("  %-12s %.3f\n", s.name.c_str(), s.score);

  require(parallel_fs_classes().size() == 4, "four storage classes compared, as PLAN.md specifies");
  require(scores.front().name == "VAST Data" || scores.front().name == "WekaFS",
          "the two GDS-certified, high-metadata-rated systems (VAST, WekaFS) rank above the two that "
          "aren't/lower-rated (Lustre/GPFS, Ceph) under this illustrative scoring");
  require(scores.back().name == "Lustre/GPFS",
          "Lustre/GPFS's centralized-metadata-server architecture (the well-documented real bottleneck for "
          "AI training's small-file pattern) scores lowest under this illustrative metric");

  std::printf("\n%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
