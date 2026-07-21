//===- prefetch_queue.h - Bounded blocking MPMC queue -------------------===//
//
// The prefetch queue between DataLoader worker threads (producers, one
// per CPU decode worker) and the training loop (consumer, pulling
// batches for the GPU). Bounded so a fast producer can't run the decode
// pipeline arbitrarily far ahead of the consumer and exhaust memory —
// capacity is the "prefetch depth" (how many samples may sit decoded-
// but-not-yet-consumed at once).
//
// This is deliberately mutex+condvar, not a lock-free queue: unlike the
// SPSC/MPMC ring buffers in foundation/ (built for hot per-element paths
// where every enqueue matters), a decode-and-push cycle here costs
// microseconds to milliseconds, so contention on one mutex is noise.
// PyTorch's DataLoader makes the same call for the same reason.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>

namespace data_loading {

template <typename T>
class PrefetchQueue {
public:
  explicit PrefetchQueue(size_t capacity) : capacity_(capacity) {}

  // Blocks while the queue is at capacity. Safe to call from multiple
  // producer threads concurrently.
  void push(T item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_full_.wait(lock, [&] { return queue_.size() < capacity_; });
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
  }

  // Blocks until an item is available or the queue is closed and
  // drained, in which case it returns nullopt (end-of-dataset signal).
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
    if (queue_.empty()) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  // Called once by the loader after every producer has finished pushing.
  // Wakes any consumer blocked in pop() with an empty queue.
  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      closed_ = true;
    }
    not_empty_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
  }

private:
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::queue<T> queue_;
  size_t capacity_;
  bool closed_ = false;
};

} // namespace data_loading
