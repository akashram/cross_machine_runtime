#pragma once

// StreamPool — pre-allocated CUDA stream pool
// =========================================================================
//
// cudaStreamCreate costs ~3–8 µs per call (driver round-trip to the CUDA
// context). Creating streams on the hot path is the wrong pattern — the
// same reason we pool cudaMalloc allocations. StreamPool creates all streams
// at construction and hands them out via RAII guards.
//
//
// Non-blocking flag
// -----------------
// All streams are created with cudaStreamNonBlocking, which means they do
// NOT synchronize implicitly with the NULL (default) stream. This is the
// right default for a pool: we want genuine concurrency between streams, not
// the accidental serialization that the default stream causes.
//
// If you need a stream that synchronizes with the default stream, pass
// cudaStreamDefault as the flags argument — but document why.
//
//
// Blocking acquire
// ----------------
// acquire() blocks (condition_variable wait) if all streams are in use.
// For Phase 3, pools are sized to never block in practice (8 streams is
// enough for all concurrent GPU operations we run). Blocking is still
// the correct fallback rather than silently serializing.
//
//
// Thread safety
// -------------
// acquire() and release() (via StreamGuard destructor) are thread-safe.
// The pool itself must outlive all StreamGuard instances it created.

#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace gpu_engine {

class StreamPool;

class StreamGuard {
public:
    StreamGuard() = default;
    ~StreamGuard() { if (pool_ && stream_) release(); }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

    StreamGuard(StreamGuard&& o) noexcept
        : pool_(o.pool_), stream_(o.stream_) {
        o.pool_ = nullptr; o.stream_ = nullptr;
    }
    StreamGuard& operator=(StreamGuard&& o) noexcept {
        if (this != &o) {
            if (pool_ && stream_) release();
            pool_ = o.pool_; stream_ = o.stream_;
            o.pool_ = nullptr; o.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get()   const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    bool valid()         const noexcept { return stream_ != nullptr; }

private:
    StreamGuard(StreamPool* pool, cudaStream_t s) : pool_(pool), stream_(s) {}
    void release();  // defined after StreamPool is complete

    friend class StreamPool;
    StreamPool*  pool_{nullptr};
    cudaStream_t stream_{nullptr};
};


class StreamPool {
public:
    explicit StreamPool(int capacity = 8,
                        unsigned int flags = cudaStreamNonBlocking) {
        streams_.reserve(capacity);
        free_.reserve(capacity);
        for (int i = 0; i < capacity; ++i) {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, flags);
            streams_.push_back(s);
            free_.push_back(s);
        }
    }

    ~StreamPool() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }

    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    StreamGuard acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !free_.empty(); });
        cudaStream_t s = free_.back();
        free_.pop_back();
        return StreamGuard(this, s);
    }

    int capacity()  const noexcept { return static_cast<int>(streams_.size()); }
    int available()       { std::lock_guard<std::mutex> lk(mu_); return static_cast<int>(free_.size()); }

private:
    void release(cudaStream_t s) {
        { std::lock_guard<std::mutex> lk(mu_); free_.push_back(s); }
        cv_.notify_one();
    }
    friend class StreamGuard;

    std::vector<cudaStream_t> streams_;
    std::vector<cudaStream_t> free_;
    std::mutex                mu_;
    std::condition_variable   cv_;
};

inline void StreamGuard::release() {
    pool_->release(stream_);
    pool_ = nullptr; stream_ = nullptr;
}

} // namespace gpu_engine
