// multitenancy_test.cpp — three tenants: "premium" (higher priority),
// "standard" and "free" (same lower priority, weights 2:1). Verifies:
// (1) every premium task executes before any standard/free task
//     (priority preemption, expressed as admission order),
// (2) standard:free execution ratio tracks their 2:1 weight,
// (3) free's small quota rejects excess submissions rather than queuing
//     them unboundedly.

#include "multitenancy.h"

#include <cstdio>
#include <vector>

int main() {
  multitenancy::FairScheduler scheduler;
  scheduler.registerTenant("premium", {/*priority=*/2, /*weight=*/1, /*quota=*/50});
  scheduler.registerTenant("standard", {/*priority=*/1, /*weight=*/2, /*quota=*/1000});
  scheduler.registerTenant("free", {/*priority=*/1, /*weight=*/1, /*quota=*/20});

  std::vector<std::pair<std::string, int>> executionOrder; // (tenant, global sequence)
  int seq = 0;
  auto makeTask = [&](const std::string &tenant) {
    return [&, tenant] { executionOrder.emplace_back(tenant, seq++); };
  };

  for (int i = 0; i < 20; ++i) scheduler.submit("premium", makeTask("premium"));
  for (int i = 0; i < 100; ++i) scheduler.submit("standard", makeTask("standard"));
  int freeAccepted = 0;
  for (int i = 0; i < 50; ++i) // submit more than free's quota (20) on purpose
    if (scheduler.submit("free", makeTask("free"))) ++freeAccepted;

  scheduler.run();

  int failures = 0;
  auto expect = [&](const char *name, bool cond) {
    std::printf("%-55s %s\n", name, cond ? "OK" : "FAIL");
    if (!cond) ++failures;
  };

  // (1) Priority: find the last premium index and first standard/free
  // index; premium must fully precede both.
  int lastPremiumSeq = -1, firstOtherSeq = INT32_MAX;
  for (auto &[tenant, s] : executionOrder) {
    if (tenant == "premium") lastPremiumSeq = std::max(lastPremiumSeq, s);
    else firstOtherSeq = std::min(firstOtherSeq, s);
  }
  expect("all premium tasks execute before any standard/free task", lastPremiumSeq < firstOtherSeq);

  // (2) Weighted fairness: standard (weight 2) should complete ~2x per
  // round relative to free (weight 1) while both still have work —
  // check via final completed counts once free (quota-limited to 20)
  // has drained: standard should have completed close to 40 by the time
  // free's last (20th) task runs, i.e. roughly weight-proportional, not
  // exactly 2x since free runs out early and standard then runs solo.
  auto standardStats = scheduler.statsFor("standard");
  auto freeStats = scheduler.statsFor("free");
  expect("standard completed all 100 submitted tasks", standardStats.completed == 100);
  expect("free completed exactly its accepted (20) tasks",
         freeStats.completed == static_cast<size_t>(freeAccepted));

  // Check the interleaving ratio directly: count standard vs free
  // completions in the prefix of executionOrder up through free's last
  // task — that's the region where both are actively contending.
  int freeLastSeq = -1;
  for (auto &[tenant, s] : executionOrder) if (tenant == "free") freeLastSeq = std::max(freeLastSeq, s);
  int standardBeforeFreeDone = 0, freeCount = 0;
  for (auto &[tenant, s] : executionOrder) {
    if (s > freeLastSeq) continue;
    if (tenant == "standard") ++standardBeforeFreeDone;
    if (tenant == "free") ++freeCount;
  }
  double ratio = static_cast<double>(standardBeforeFreeDone) / freeCount;
  std::printf("  (standard:free completions while both contending = %d:%d, ratio %.2f, weight ratio is 2.0)\n",
              standardBeforeFreeDone, freeCount, ratio);
  expect("standard:free contended ratio is close to weight ratio (2.0, +/-0.5)", ratio > 1.5 && ratio < 2.5);

  // (3) Quota enforcement.
  expect("free accepted exactly its quota (20) of 50 submitted", freeAccepted == 20);
  expect("free rejected exactly 30", freeStats.rejected == 30);

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
