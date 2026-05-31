#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>

namespace foundation {

// Single-producer single-consumer lock-free ring buffer.
//
// Constraints:
//   T must be trivially copyable. The queue copies values directly into
//   slots without invoking constructors or destructors on the slot storage.
//
//   N must be a power of two >= 2. Modular indexing uses (index & mask)
//   instead of (index % N) — avoids division on every push/pop.
//
//   Effective capacity is N-1. One slot is sacrificed as the sentinel that
//   distinguishes "full" (next_tail == head) from "empty" (tail == head).
//
// Ownership model:
//   tail_ is owned by the producer — only the producer writes it.
//   head_ is owned by the consumer — only the consumer writes it.
//   Each thread reads its own index with relaxed because no other thread
//   competes on writes to it. Each thread reads the other thread's index
//   with acquire, paired with the other thread's release store, to establish
//   a happens-before edge across the shared slot.
//
// Memory ordering on x86 (TSO):
//   acquire loads and release stores compile to plain MOV — no fence
//   instruction is emitted. The annotations are needed for correctness on
//   ARM (where loads can pass loads, and stores can pass stores) and to
//   communicate intent to the compiler's optimizer.
//
// False sharing:
//   head_ and tail_ are on separate cache lines. Without this, every push
//   would dirty the consumer's cache line (which contains head_) and every
//   pop would dirty the producer's cache line (which contains tail_),
//   causing coherence traffic proportional to throughput even though the
//   two threads never contend on the same variable.

template <typename T, std::size_t N>
class SpscQueue {
    static_assert(std::is_trivially_copyable_v<T>,
        "SpscQueue<T>: T must be trivially copyable");
    static_assert(N >= 2 && (N & (N - 1)) == 0,
        "SpscQueue<N>: N must be a power of two >= 2");

    static constexpr std::size_t kMask = N - 1;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) T slots_[N];

public:
    SpscQueue() = default;
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    // Push one item. Returns false if the queue is full.
    // Must be called from the producer thread only.
    bool push(const T& val) noexcept {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        std::size_t next = (tail + 1) & kMask;
        // Acquire: synchronizes with the consumer's release store to head_.
        // Ensures we see all slots that the consumer has freed.
        if (next == head_.load(std::memory_order_acquire))
            return false;
        slots_[tail] = val;
        // Release: publishes the slot write to the consumer.
        tail_.store(next, std::memory_order_release);
        return true;
    }

    // Pop one item. Returns false if the queue is empty.
    // Must be called from the consumer thread only.
    bool pop(T& val) noexcept {
        std::size_t head = head_.load(std::memory_order_relaxed);
        // Acquire: synchronizes with the producer's release store to tail_.
        // Ensures we see the slot data the producer wrote before its tail store.
        if (head == tail_.load(std::memory_order_acquire))
            return false;
        val = slots_[head];
        // Release: publishes the freed slot to the producer.
        head_.store((head + 1) & kMask, std::memory_order_release);
        return true;
    }

    // Returns true if the queue appeared empty at the moment of the call.
    // Stale by the time the caller acts on the result.
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    // Approximate item count. Can be stale immediately after the call.
    std::size_t size_approx() const noexcept {
        std::size_t head = head_.load(std::memory_order_acquire);
        std::size_t tail = tail_.load(std::memory_order_acquire);
        return (tail - head + N) & kMask;
    }

    // Maximum number of items the queue can hold simultaneously.
    static constexpr std::size_t capacity() noexcept { return N - 1; }
};

} // namespace foundation
