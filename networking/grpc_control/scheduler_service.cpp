//===- scheduler_service.cpp - Scheduler gRPC service implementation ----===//
//
// WaitTask blocks (with a condition variable, not polling) until
// completeTask() is called for that task id — chosen over a polling loop
// because the latency benchmark this step's README wants (p50/p99 RPC
// latency) would otherwise be measuring polling-interval jitter, not the
// actual dispatch-to-completion latency.
//
//===----------------------------------------------------------------------===//

#include "scheduler_service.h"

namespace runtime {

std::string SchedulerServiceImpl::newId(const char *prefix) {
  return std::string(prefix) + "-" + std::to_string(nextId_.fetch_add(1));
}

grpc::Status SchedulerServiceImpl::RegisterNode(grpc::ServerContext *, const NodeInfo *info,
                                                 NodeId *id) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string nodeId = newId("node");
  NodeRecord record;
  record.hostname = info->hostname();
  record.devices.assign(info->devices().begin(), info->devices().end());
  nodes_.emplace(nodeId, std::move(record));
  id->set_id(nodeId);
  return grpc::Status::OK;
}

grpc::Status SchedulerServiceImpl::HealthCheck(grpc::ServerContext *, const NodeId *id,
                                                HealthStatus *status) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = nodes_.find(id->id());
  if (it == nodes_.end()) {
    status->set_healthy(false);
    status->set_message("unknown node id: " + id->id());
    return grpc::Status::OK;
  }
  status->set_healthy(it->second.healthy);
  status->set_message(it->second.healthy ? "ok" : "marked unhealthy");
  return grpc::Status::OK;
}

grpc::Status SchedulerServiceImpl::DispatchTask(grpc::ServerContext *, const TaskSpec *spec,
                                                 TaskId *id) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string taskId = newId("task");
  pendingTasks_.emplace(taskId, *spec);
  id->set_id(taskId);
  // Actual dispatch-to-a-worker is out of scope here (no worker-facing
  // RPC in this .proto — see scheduler_service.h's completeTask comment);
  // this records the task as pending so WaitTask has something to block
  // on and a test harness can call completeTask() directly.
  return grpc::Status::OK;
}

grpc::Status SchedulerServiceImpl::WaitTask(grpc::ServerContext *, const TaskId *id,
                                             TaskResult *result) {
  std::unique_lock<std::mutex> lock(mu_);
  taskDone_.wait(lock, [&] { return completedTasks_.count(id->id()) > 0; });
  *result = completedTasks_.at(id->id());
  return grpc::Status::OK;
}

void SchedulerServiceImpl::completeTask(const std::string &task_id, TaskResult result) {
  std::lock_guard<std::mutex> lock(mu_);
  pendingTasks_.erase(task_id);
  completedTasks_[task_id] = std::move(result);
  taskDone_.notify_all();
}

} // namespace runtime
