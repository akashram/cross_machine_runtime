//===- scheduler_service.h - Scheduler gRPC service implementation ------===//
#pragma once

#include "scheduler.grpc.pb.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace runtime {

// In-memory implementation — no persistence, no leader election. This is
// the control plane's RPC surface; making the underlying state durable
// and replicated is Raft's job (step 19), layered underneath this service
// rather than reimplemented here.
class SchedulerServiceImpl final : public Scheduler::Service {
public:
  grpc::Status RegisterNode(grpc::ServerContext *ctx, const NodeInfo *info, NodeId *id) override;
  grpc::Status HealthCheck(grpc::ServerContext *ctx, const NodeId *id, HealthStatus *status) override;
  grpc::Status DispatchTask(grpc::ServerContext *ctx, const TaskSpec *spec, TaskId *id) override;
  grpc::Status WaitTask(grpc::ServerContext *ctx, const TaskId *id, TaskResult *result) override;

  // Test/worker-side hook: mark a dispatched task complete. In a real
  // deployment this is called from the worker-side RPC client after it
  // finishes running the IR; there's no "push" path from worker to
  // scheduler in this service definition (WaitTask is scheduler-polled by
  // the caller, not pushed to), so this is the seam a worker-facing
  // service (not defined here — out of scope for step 6) would call into.
  void completeTask(const std::string &task_id, TaskResult result);

private:
  struct NodeRecord {
    std::string hostname;
    std::vector<std::string> devices;
    bool healthy = true;
  };

  std::mutex mu_;
  std::condition_variable taskDone_;
  std::atomic<uint64_t> nextId_{0};
  std::unordered_map<std::string, NodeRecord> nodes_;
  std::unordered_map<std::string, TaskResult> completedTasks_;
  std::unordered_map<std::string, TaskSpec> pendingTasks_;

  std::string newId(const char *prefix);
};

} // namespace runtime
