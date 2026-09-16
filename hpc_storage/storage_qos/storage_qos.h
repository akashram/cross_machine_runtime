//===- storage_qos.h - Storage bandwidth/IOPS QoS, built on multitenancy ===//
//
// PLAN.md Phase 21 step 8: "direct extension of networking/multitenancy's
// existing fairness/quota logic, reframed for storage bandwidth/IOPS
// contention between simultaneous training jobs sharing one backend,
// rather than network bandwidth between peers -- same mechanism,
// different contended resource."
//
// This is a genuine extension, not a rewrite: `networking::multitenancy`'s
// FairScheduler already solves priority-then-weighted-round-robin task
// ADMISSION with a per-tenant task-COUNT quota. Storage QoS needs the same
// priority/fairness admission policy but a per-tenant BYTES/SEC quota
// instead (bandwidth, not task count, is the contended resource for
// storage I/O). Rather than duplicate FairScheduler's admission logic,
// StorageQosScheduler composes an unmodified FairScheduler for priority/
// weight ordering and adds a token-bucket bandwidth limiter per tenant on
// top -- the same "same mechanism, different contended resource"
// composition PLAN.md asks for.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../../networking/multitenancy/multitenancy.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace hpc_storage {

struct StorageTenantConfig {
  int priority;
  int weight;
  size_t quota;               // max queued I/O requests (multitenancy::TenantConfig::quota, unchanged meaning)
  double bandwidth_bytes_sec; // this tenant's fair-share storage bandwidth budget
};

struct StorageIoStats {
  size_t requests_submitted = 0;
  size_t requests_rejected_quota = 0;    // rejected by FairScheduler's task-count quota (unchanged mechanism)
  size_t bytes_admitted = 0;             // bytes actually let through within the bandwidth budget
  size_t bytes_throttled = 0;            // bytes that exceeded the token-bucket budget and were NOT admitted
};

// A token bucket per tenant: `bytes_available` refills at
// `bandwidth_bytes_sec` per simulated second and is spent by admit(). This
// is the storage-specific addition multitenancy.h has no equivalent of
// (its quota is a task COUNT, not a byte rate) -- everything else
// (priority ordering, weighted round-robin admission) is reused from
// multitenancy::FairScheduler exactly as-is.
class StorageQosScheduler {
public:
  void registerTenant(const std::string &name, StorageTenantConfig config) {
    configs_[name] = config;
    buckets_[name] = config.bandwidth_bytes_sec; // start each tenant with one second's worth of budget
    scheduler_.registerTenant(name, multitenancy::TenantConfig{config.priority, config.weight, config.quota});
  }

  // Advances simulated time by `elapsed_sec`, refilling every tenant's
  // token bucket (capped at one second's worth, so idle tenants can't
  // bank unbounded burst credit -- standard token-bucket discipline).
  void tick(double elapsed_sec) {
    for (auto &kv : buckets_) {
      double cap = configs_.at(kv.first).bandwidth_bytes_sec;
      kv.second = std::min(cap, kv.second + cap * elapsed_sec);
    }
  }

  // Attempts to admit one I/O request of `bytes`. Two independent gates,
  // both reused/composed rather than reimplemented: (1) FairScheduler's
  // own task-count quota (submit() returns false if exceeded -- the
  // EXACT unmodified multitenancy mechanism), (2) this class's
  // bandwidth token bucket. A request only counts as admitted if BOTH
  // gates pass.
  bool submit(const std::string &tenant, size_t bytes) {
    bool queued = scheduler_.submit(tenant, [] {}); // no-op task; FairScheduler's admission gate is what matters
    StorageIoStats &s = stats_[tenant];
    s.requests_submitted++;
    if (!queued) {
      s.requests_rejected_quota++;
      return false;
    }
    double &bucket = buckets_.at(tenant);
    if (static_cast<double>(bytes) <= bucket) {
      bucket -= static_cast<double>(bytes);
      s.bytes_admitted += bytes;
      return true;
    }
    s.bytes_throttled += bytes;
    return false;
  }

  // Drains FairScheduler's admitted-request queue in priority/weighted
  // round-robin order (the real unmodified FairScheduler::run()) --
  // models the storage system actually servicing its queued, already
  // bandwidth-checked requests. Call periodically (e.g. once per tick())
  // so a tenant's task-count quota frees up for the next round.
  void drain() { scheduler_.run(); }

  StorageIoStats statsFor(const std::string &tenant) const { return stats_.at(tenant); }

private:
  multitenancy::FairScheduler scheduler_;
  std::unordered_map<std::string, StorageTenantConfig> configs_;
  std::unordered_map<std::string, double> buckets_;
  std::unordered_map<std::string, StorageIoStats> stats_;
};

} // namespace hpc_storage
