#pragma once

// WorkStealingPool: fixed-size thread pool with per-worker Chase-Lev deques.
//
// Each worker thread owns a ChaseLevDeque<Task*>:
//   - The worker pushes/pops from its own deque (LIFO — keeps hot tasks local).
//   - Idle workers steal from peers' deques (FIFO — oldest tasks first).
//
// Submission routing:
//   submit() called from a WORKER thread → push to that worker's own deque.
//     This preserves task locality: child tasks spawned by a running task land
//     next to their parent in the LIFO stack, maximising cache reuse.
//   submit() called from an EXTERNAL thread → push to the shared inbox.
//     Workers drain the inbox as part of their loop.
//
// Sleep/wake:
//   Workers spin for kSpinCount iterations before sleeping on a condition
//   variable. The inbox condvar wakes one sleeping worker per submission.
//   A 1 ms timeout ensures workers drain deques even if they missed a notify.
//
// Task graph (TaskGroup):
//   TaskGroup tracks a set of tasks submitted to the pool and lets callers
//   wait until all of them complete. It uses an atomic counter:
//     run(fn) → remaining_++, submit wrapper that decrements on completion
//     wait()  → blocks until remaining_ == 0
//
//   This is the foundation for parallel_for and divide-and-conquer patterns.
//   Arbitrary DAG dependencies (step 13, coroutine tasks) build on top of this.
//
// Shutdown:
//   The destructor sets shutdown_, wakes all workers, and joins threads.
//   Callers must ensure no tasks are in flight (call wait() first) before
//   destroying the pool — destroying with pending tasks is undefined.

#include "chase_lev/chase_lev.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace foundation {

// ---------------------------------------------------------------------------
// Forward declaration
// ---------------------------------------------------------------------------
class WorkStealingPool;

// ---------------------------------------------------------------------------
// TaskGroup — wait for a batch of tasks submitted to the pool
// ---------------------------------------------------------------------------
class TaskGroup {
public:
    explicit TaskGroup(WorkStealingPool& pool) noexcept : pool_(pool) {}

    TaskGroup(const TaskGroup&)            = delete;
    TaskGroup& operator=(const TaskGroup&) = delete;

    // Submit fn to the pool; wait() will not return until fn has completed.
    void run(std::function<void()> fn);

    // Block until all run()-submitted tasks have completed.
    void wait() {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [this]{
            return remaining_.load(std::memory_order_acquire) == 0;
        });
    }

private:
    WorkStealingPool&       pool_;
    std::atomic<int>        remaining_{0};
    std::mutex              mu_;
    std::condition_variable cv_;
};

// ---------------------------------------------------------------------------
// WorkStealingPool
// ---------------------------------------------------------------------------
class WorkStealingPool {
public:
    explicit WorkStealingPool(
        std::size_t n_threads = std::thread::hardware_concurrency());

    // Waits for all in-flight tasks to complete, then joins workers.
    ~WorkStealingPool() noexcept;

    WorkStealingPool(const WorkStealingPool&)            = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    // Submit a callable for async execution.
    // Thread-safe. If called from a worker thread, the task goes directly to
    // that worker's Chase-Lev deque. Otherwise it enters the shared inbox.
    void submit(std::function<void()> fn);

    // Run fn(0), fn(1), ..., fn(n-1) in parallel and block until all done.
    void parallel_for(std::size_t n, std::function<void(std::size_t)> fn);

    // Block until all pending tasks submitted via submit() have completed.
    void wait();

    std::size_t thread_count() const noexcept { return workers_.size(); }

private:
    friend class TaskGroup;

    // -----------------------------------------------------------------------
    // Internal task node
    // -----------------------------------------------------------------------
    struct Task {
        std::function<void()> fn;
    };

    // -----------------------------------------------------------------------
    // Per-worker state, cache-line aligned to avoid false sharing between
    // adjacent workers' deques.
    // -----------------------------------------------------------------------
    struct alignas(64) Worker {
        ChaseLevDeque<Task*> deque;
        std::thread          thread;
    };

    // -----------------------------------------------------------------------
    // Thread-local accessors (function-local thread_local = ODR-safe)
    // -----------------------------------------------------------------------
    static WorkStealingPool*& tl_pool() noexcept {
        thread_local WorkStealingPool* p = nullptr;
        return p;
    }
    static std::size_t& tl_idx() noexcept {
        thread_local std::size_t i = ~std::size_t{0};
        return i;
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------
    void   worker_loop(std::size_t idx) noexcept;
    Task*  try_steal(std::size_t self_idx) noexcept;
    void   execute(Task* task) noexcept;

    static constexpr int kSpinCount = 200;

    std::vector<std::unique_ptr<Worker>> workers_;

    // Shared inbox for external (non-worker) submissions.
    std::mutex              inbox_mu_;
    std::vector<Task*>      inbox_;
    std::condition_variable inbox_cv_;

    // pending_: counts tasks submitted but not yet completed.
    std::atomic<std::size_t> pending_{0};
    std::mutex               done_mu_;
    std::condition_variable  done_cv_;

    std::atomic<bool> shutdown_{false};
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------

inline WorkStealingPool::WorkStealingPool(std::size_t n_threads) {
    if (n_threads == 0) n_threads = 1;
    workers_.reserve(n_threads);

    // Phase 1: allocate all Worker structs so workers_ is fully sized before
    // any thread starts. Worker threads read workers_.size() via try_steal();
    // if we interleaved alloc + spawn, early threads would race on the vector.
    for (std::size_t i = 0; i < n_threads; ++i)
        workers_.push_back(std::make_unique<Worker>());

    // Phase 2: spawn threads. workers_ is now stable (read-only from here on).
    for (std::size_t i = 0; i < n_threads; ++i)
        workers_[i]->thread = std::thread([this, i]{ worker_loop(i); });
}

inline WorkStealingPool::~WorkStealingPool() noexcept {
    shutdown_.store(true, std::memory_order_release);
    inbox_cv_.notify_all();
    for (auto& w : workers_) w->thread.join();
}

inline void WorkStealingPool::submit(std::function<void()> fn) {
    auto* task = new Task{std::move(fn)};
    pending_.fetch_add(1, std::memory_order_relaxed);

    if (tl_pool() == this) {
        // Worker thread: push to own Chase-Lev deque (LIFO locality).
        workers_[tl_idx()]->deque.push(task);
    } else {
        // External thread: push to shared inbox.
        {
            std::lock_guard lk(inbox_mu_);
            inbox_.push_back(task);
        }
        inbox_cv_.notify_one();
    }
}

inline void WorkStealingPool::parallel_for(
        std::size_t n, std::function<void(std::size_t)> fn) {
    TaskGroup group(*this);
    for (std::size_t i = 0; i < n; ++i)
        group.run([fn, i]{ fn(i); });
    group.wait();
}

inline void WorkStealingPool::wait() {
    std::unique_lock lk(done_mu_);
    done_cv_.wait(lk, [this]{
        return pending_.load(std::memory_order_acquire) == 0;
    });
}

inline void WorkStealingPool::execute(Task* task) noexcept {
    task->fn();
    delete task;
    // Hold done_mu_ when notifying to prevent the lost-wakeup race: without
    // the lock, notify_all() can arrive between wait()'s predicate returning
    // false and the thread actually blocking on the condvar, causing an
    // indefinite sleep.
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard lk(done_mu_);
        done_cv_.notify_all();
    }
}

inline WorkStealingPool::Task* WorkStealingPool::try_steal(std::size_t self) noexcept {
    std::size_t n = workers_.size();
    for (std::size_t k = 1; k < n; ++k) {
        std::size_t victim = (self + k) % n;
        if (auto v = workers_[victim]->deque.steal())
            return *v;
    }
    return nullptr;
}

inline void WorkStealingPool::worker_loop(std::size_t idx) noexcept {
    tl_pool() = this;
    tl_idx()  = idx;

    int spin = 0;

    while (!shutdown_.load(std::memory_order_acquire)) {
        Task* task = nullptr;

        // 1. Own deque — LIFO, no CAS.
        if (auto v = workers_[idx]->deque.pop())
            task = *v;

        // 2. Shared inbox — grab one task (try_lock avoids blocking).
        if (!task) {
            std::unique_lock lk(inbox_mu_, std::try_to_lock);
            if (lk.owns_lock() && !inbox_.empty()) {
                task = inbox_.back();
                inbox_.pop_back();
            }
        }

        // 3. Steal from a peer — FIFO from their deque top.
        if (!task)
            task = try_steal(idx);

        if (task) {
            spin = 0;
            execute(task);
            continue;
        }

        // No work found — spin briefly, then sleep.
        if (++spin < kSpinCount) {
            std::this_thread::yield();
            continue;
        }
        spin = 0;

        std::unique_lock lk(inbox_mu_);
        inbox_cv_.wait_for(lk, std::chrono::milliseconds(1), [this]{
            return !inbox_.empty() || shutdown_.load(std::memory_order_relaxed);
        });
    }

    // Drain anything remaining after shutdown signal.
    while (true) {
        Task* task = nullptr;
        if (auto v = workers_[idx]->deque.pop())  task = *v;
        if (!task) {
            std::lock_guard lk(inbox_mu_);
            if (!inbox_.empty()) { task = inbox_.back(); inbox_.pop_back(); }
        }
        if (!task) break;
        execute(task);
    }
}

// ---------------------------------------------------------------------------
// TaskGroup::run — defined after WorkStealingPool is complete
// ---------------------------------------------------------------------------
inline void TaskGroup::run(std::function<void()> fn) {
    remaining_.fetch_add(1, std::memory_order_relaxed);
    pool_.submit([this, fn = std::move(fn)]() mutable {
        fn();
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard lk(mu_);
            cv_.notify_all();
        }
    });
}

} // namespace foundation
