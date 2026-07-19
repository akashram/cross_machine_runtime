//===- multitenancy.h - Resource quotas + priority + fair scheduling ----===//
//
// Portable — pure scheduling logic, no network dependency.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace multitenancy {

struct TenantConfig {
  int priority;    // higher runs first; strict ordering across priority classes
  int weight;      // relative share within the same priority class (round-robin quantum)
  size_t quota;     // max tasks this tenant may have submit()ted and not-yet-run at once
};

struct TenantStats {
  size_t submitted = 0;
  size_t rejected = 0;  // quota exceeded at submit time
  size_t completed = 0;
};

// Single-threaded, run-to-completion scheduler: submit() enqueues (or
// rejects on quota overflow), run() drains every queued task in
// priority-then-weighted-round-robin order. Not a live server (no
// concurrent submit-while-running) — this is the scheduling *policy* in
// isolation, the same scope discipline as topo_scheduler's placement
// algorithm (step 16): validate the algorithm's fairness/priority
// properties directly, independent of however a real service would wire
// concurrent submission and execution around it.
class FairScheduler {
public:
  void registerTenant(const std::string &name, TenantConfig config);

  // Returns false (and increments the tenant's rejected count) if the
  // tenant already has `quota` tasks queued.
  bool submit(const std::string &tenant, std::function<void()> task);

  // Runs every queued task to completion, priority class by priority
  // class (highest first — all of a higher class drains before any
  // lower class task runs), weighted round-robin within a class.
  void run();

  TenantStats statsFor(const std::string &tenant) const;

private:
  struct TenantQueue {
    TenantConfig config;
    std::queue<std::function<void()>> tasks;
    TenantStats stats;
  };

  std::unordered_map<std::string, TenantQueue> tenants_;
  std::vector<std::string> order_; // registration order, for deterministic round-robin
};

} // namespace multitenancy
