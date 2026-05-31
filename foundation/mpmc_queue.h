#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace foundation {

// Multi-producer multi-consumer lock-free ring buffer.
//
// Algorithm: Dmitry Vyukov's sequence-number MPMC queue (2008).
// Each slot carries an atomic sequence number that acts as a per-slot
// generation counter. Producers CAS on tail_ to claim a slot; consumers
// CAS on head_ to claim an item. The sequence number — not the position
// counter — is what establishes happens-before between writer and reader.
//
// Capacity is exactly N (no wasted sentinel slot). Full and empty are
// detected by comparing the slot's sequence against the expected value.
//
// Sequence number protocol:
//   Initial state:   slots_[i].sequence = i  for i in [0, N)
//   After push at position pos:  sequence = pos + 1       (slot has data)
//   After pop  at position pos:  sequence = pos + N       (slot is free)
//
//   Producer checks: seq == pos       (slot is free, ready to write)
//   Consumer checks: seq == pos + 1   (slot has data, ready to read)
//
//   dif < 0: queue is full (producer) or empty (consumer) — return false
//   dif > 0: our cached pos is stale; reload and retry
//   dif == 0 + CAS succeeds: we own the slot
//
// ABA analysis:
//   The classic ABA problem arises when a CAS target (typically a pointer)
//   is freed, reallocated at the same address, and a stale CAS incorrectly
//   succeeds. Here, the CAS targets are tail_ and head_ — monotonically
//   increasing position counters, not pointers. Wraparound (after 2^64
//   operations) is the only theoretical source of ABA. At 1 billion ops/sec
//   that is ~585 years. Crucially, even if wraparound occurred, the per-slot
//   sequence number provides a second guard: a slot whose sequence doesn't
//   match the expected value will not be claimed, so an incorrectly succeeding
//   CAS on the position counter would immediately fail the sequence check and
//   retry. ABA is therefore doubly eliminated: by the counter width and by the
//   per-slot generation counter.
//
// False sharing:
//   tail_ and head_ are on separate cache lines (same reasoning as SpscQueue).
//   Additionally, each Slot is padded to a full cache line. Without per-slot
//   padding, multiple producers writing to adjacent slots simultaneously would
//   cause coherence traffic even though they own different slots — they'd
//   share the same cache line, causing the line to bounce between cores on
//   every write.
//
// Overhead vs. SpscQueue:
//   In the 1P-1C case, MpmcQueue is slower than SpscQueue because:
//   1. Every push and pop performs a CAS (not just a store) on the position.
//   2. tail_ and head_ are shared among all producers/consumers respectively,
//      so they bounce between cores under contention.
//   3. The sequence check adds an extra acquire load per operation.
//   Use SpscQueue when you have exactly one producer and one consumer.
//
// T must be trivially copyable. N must be a power of two >= 2.

template <typename T, std::size_t N>
class MpmcQueue {
    static_assert(std::is_trivially_copyable_v<T>,
        "MpmcQueue<T>: T must be trivially copyable");
    static_assert(N >= 2 && (N & (N - 1)) == 0,
        "MpmcQueue<N>: N must be a power of two >= 2");

    static constexpr std::size_t kMask = N - 1;

    // Each slot is a full cache line. Without this padding, producers writing
    // adjacent slots simultaneously share a cache line, causing false sharing
    // coherence traffic proportional to the number of concurrent producers.
    struct alignas(64) Slot {
        std::atomic<std::size_t> sequence;
        T data;
    };

    // tail_ is the hot cache line under multiple producers: every push does a
    // CAS on it. head_ is the hot cache line under multiple consumers. Keeping
    // them on separate lines prevents the producer and consumer sides from
    // dirtying each other's working set.
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::atomic<std::size_t> head_{0};
    Slot slots_[N];

public:
    MpmcQueue() noexcept {
        for (std::size_t i = 0; i < N; ++i)
            slots_[i].sequence.store(i, std::memory_order_relaxed);
    }

    MpmcQueue(const MpmcQueue&) = delete;
    MpmcQueue& operator=(const MpmcQueue&) = delete;

    // Push one item. Returns false if the queue is full.
    // Safe to call from any number of concurrent producer threads.
    bool push(const T& val) noexcept {
        std::size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[pos & kMask];
            // Acquire: synchronizes with the consumer's release store that
            // marked this slot free (sequence = pos + N from a prior cycle,
            // or sequence = pos on the very first cycle).
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            // Cast to signed to correctly detect both "full" (dif < 0) and
            // "stale pos" (dif > 0) even when values are near size_t max.
            std::intptr_t dif =
                static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);

            if (dif == 0) {
                // Slot is free and matches our position. Race to claim it.
                // On CAS failure, pos is updated to the current tail_ value.
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed))
                    break;
            } else if (dif < 0) {
                return false;  // queue is full
            } else {
                // pos is stale — another producer advanced tail_ past it.
                // Reload and retry. (CAS would also fail and reload, but this
                // avoids the unnecessary CAS attempt.)
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        slots_[pos & kMask].data = val;
        // Release: publishes the data write to any consumer that subsequently
        // does an acquire load of sequence.
        slots_[pos & kMask].sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Pop one item. Returns false if the queue is empty.
    // Safe to call from any number of concurrent consumer threads.
    bool pop(T& val) noexcept {
        std::size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[pos & kMask];
            // Acquire: synchronizes with the producer's release store of
            // sequence = pos + 1, ensuring the data write is visible.
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            std::intptr_t dif =
                static_cast<std::intptr_t>(seq) -
                static_cast<std::intptr_t>(pos + 1);

            if (dif == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed))
                    break;
            } else if (dif < 0) {
                return false;  // queue is empty
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        val = slots_[pos & kMask].data;
        // Release: marks the slot as free for producers (sequence = pos + N
        // is exactly what the producer will expect when this slot comes around
        // again after N more positions).
        slots_[pos & kMask].sequence.store(pos + N, std::memory_order_release);
        return true;
    }

    // Maximum number of items the queue can hold simultaneously.
    // Unlike SpscQueue, MpmcQueue does not waste a sentinel slot — capacity is N.
    static constexpr std::size_t capacity() noexcept { return N; }
};

} // namespace foundation
