#pragma once

// EventPool — pre-allocated CUDA event pool
// =========================================================================
//
// cudaEventCreate costs ~2–5 µs. For any code path that creates events per
// operation this is meaningful overhead. EventPool amortises the cost by
// allocating all events upfront.
//
//
// Timing vs sync events
// ---------------------
// cudaEventDefault         — records a timestamp; needed for cudaEventElapsedTime.
//                            Has a small overhead on Record and Synchronize.
// cudaEventDisableTiming   — no timestamp; cheaper to record and wait on.
//                            Use this for dependency injection (stream A waits
//                            for stream B to reach a point) where you don't
//                            need the elapsed time.
//
// Default: cudaEventDisableTiming because most events in the stream manager
// are used for cross-stream synchronization, not timing. Create a separate
// EventPool with cudaEventDefault for benchmarking.

#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace gpu_engine {

class EventPool;

class EventGuard {
public:
    EventGuard() = default;
    ~EventGuard() { if (pool_ && event_) release(); }

    EventGuard(const EventGuard&) = delete;
    EventGuard& operator=(const EventGuard&) = delete;

    EventGuard(EventGuard&& o) noexcept
        : pool_(o.pool_), event_(o.event_) {
        o.pool_ = nullptr; o.event_ = nullptr;
    }
    EventGuard& operator=(EventGuard&& o) noexcept {
        if (this != &o) {
            if (pool_ && event_) release();
            pool_ = o.pool_; event_ = o.event_;
            o.pool_ = nullptr; o.event_ = nullptr;
        }
        return *this;
    }

    cudaEvent_t get()   const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }
    bool valid()         const noexcept { return event_ != nullptr; }

private:
    EventGuard(EventPool* pool, cudaEvent_t e) : pool_(pool), event_(e) {}
    void release();

    friend class EventPool;
    EventPool*  pool_{nullptr};
    cudaEvent_t event_{nullptr};
};


class EventPool {
public:
    explicit EventPool(int capacity = 32,
                       unsigned int flags = cudaEventDisableTiming) {
        events_.reserve(capacity);
        free_.reserve(capacity);
        for (int i = 0; i < capacity; ++i) {
            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, flags);
            events_.push_back(e);
            free_.push_back(e);
        }
    }

    ~EventPool() {
        for (auto e : events_) cudaEventDestroy(e);
    }

    EventPool(const EventPool&) = delete;
    EventPool& operator=(const EventPool&) = delete;

    EventGuard acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !free_.empty(); });
        cudaEvent_t e = free_.back();
        free_.pop_back();
        return EventGuard(this, e);
    }

    int capacity()  const noexcept { return static_cast<int>(events_.size()); }
    int available()       { std::lock_guard<std::mutex> lk(mu_); return static_cast<int>(free_.size()); }

private:
    void release(cudaEvent_t e) {
        { std::lock_guard<std::mutex> lk(mu_); free_.push_back(e); }
        cv_.notify_one();
    }
    friend class EventGuard;

    std::vector<cudaEvent_t> events_;
    std::vector<cudaEvent_t> free_;
    std::mutex               mu_;
    std::condition_variable  cv_;
};

inline void EventGuard::release() {
    pool_->release(event_);
    pool_ = nullptr; event_ = nullptr;
}

} // namespace gpu_engine
