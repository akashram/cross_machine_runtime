#include "multitenancy.h"

#include <algorithm>
#include <set>

namespace multitenancy {

void FairScheduler::registerTenant(const std::string &name, TenantConfig config) {
  tenants_[name] = TenantQueue{config, {}, {}};
  order_.push_back(name);
}

bool FairScheduler::submit(const std::string &tenant, std::function<void()> task) {
  TenantQueue &tq = tenants_.at(tenant);
  if (tq.tasks.size() >= tq.config.quota) {
    tq.stats.rejected++;
    return false;
  }
  tq.tasks.push(std::move(task));
  tq.stats.submitted++;
  return true;
}

void FairScheduler::run() {
  std::set<int, std::greater<int>> priorities; // descending: highest priority class first
  for (const auto &name : order_) priorities.insert(tenants_[name].config.priority);

  for (int prio : priorities) {
    std::vector<std::string> group;
    for (const auto &name : order_)
      if (tenants_[name].config.priority == prio) group.push_back(name);

    // Weighted round-robin within this priority class: every tenant in
    // the class gets `weight` tasks per pass, repeated until the whole
    // class's queues are drained. All of this class runs to completion
    // before the next (lower) priority class gets to submit a single
    // task — the "priority preemption" this step is about, expressed as
    // admission order rather than mid-task interruption (tasks here are
    // opaque callables with no natural suspend point).
    for (;;) {
      bool anyRemaining = std::any_of(group.begin(), group.end(), [&](const std::string &name) {
        return !tenants_[name].tasks.empty();
      });
      if (!anyRemaining) break;

      for (const auto &name : group) {
        TenantQueue &tq = tenants_[name];
        for (int i = 0; i < tq.config.weight && !tq.tasks.empty(); ++i) {
          auto task = std::move(tq.tasks.front());
          tq.tasks.pop();
          task();
          tq.stats.completed++;
        }
      }
    }
  }
}

TenantStats FairScheduler::statsFor(const std::string &tenant) const {
  return tenants_.at(tenant).stats;
}

} // namespace multitenancy
