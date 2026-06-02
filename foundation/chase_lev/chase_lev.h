#pragma once

// ChaseLevDeque<T>: lock-free work-stealing double-ended queue, header-only.
//
// Papers:
//   Chase & Lev, "Dynamic Circular Work-Stealing Deque," SPAA 2005.
//   Lê, Pop, Liu, Nardelli, "Correct and Efficient Work-Stealing for Weak
//   Memory Models," PPoPP 2013.  (fixes memory ordering for ARM/POWER)
//
// Asymmetry — the central design insight:
//   One designated OWNER thread pushes and pops from the BOTTOM (the "local"
//   end). This is the hot path for a worker thread processing its own task
//   queue: push a new task, pop to execute immediately (LIFO scheduling).
//   Push and pop require NO CAS in the common case (uncontended owner).
//
//   ANY thread may steal from the TOP (the "remote" end). Steal is the cold
//   path: only used when a thread runs out of work and needs to take from a
//   peer. Steal does one CAS.
//
//   This asymmetry is what makes work-stealing efficient. If all N threads
//   work on their own tasks, the CAS contention is zero.
//
// Layout:
//   [top .......... bottom)   — live tasks, FIFO from top, LIFO from bottom
//   [0              capacity) — circular buffer (capacity = power of 2)
//
//   top_   (int64): only written by thieves (CAS)
//   bottom_(int64): only written by the owner
//   array_ (Array*): updated only by the owner on growth; read by all
//
// Owner push (the fast path):
//   1. If buffer full: grow (2× capacity, copy elements, keep old array).
//   2. data[b % cap].store(val, relaxed)
//   3. bottom_.store(b+1, release)  — release makes data visible to stealers
//   No CAS; no locks. The release store on bottom synchronizes with steal()'s
//   acquire load of bottom, ensuring data[b] is visible to any thief that
//   reads bottom = b+1. This is semantically equivalent to the original
//   fence(release) + relaxed-bottom pattern from the Lê et al. paper, but
//   expressed as a single release store so TSan can model the synchronization.
//
// Owner pop — LIFO, opposite end from steal:
//   1. bottom_.store(b-1, relaxed)     — tentatively claim the last slot
//   2. fence(seq_cst)                  — dual fence with steal's fence (below)
//   3. t = top_.load(relaxed)
//   4. If t > b-1: empty (thief took it). Restore bottom. Return nullopt.
//   5. val = data[(b-1) % cap].load(relaxed)
//   6. If t == b-1: exactly one element — race with concurrent steal.
//      CAS(top, t, t+1, seq_cst). If CAS fails: lost to thief. Restore. Return nullopt.
//   7. Return val.
//
// Steal — FIFO, any thread:
//   1. t = top_.load(acquire)
//   2. fence(seq_cst)                  — see below
//   3. b = bottom_.load(acquire)
//   4. If t >= b: empty.
//   5. val = data[t % cap].load(relaxed)   — safe via acquire on bottom (see below)
//   6. CAS(top, t, t+1, seq_cst). If fails: another thread stole. Return nullopt.
//   7. Return val.
//
// Why dual seq_cst fences in pop() and steal()?
//   When exactly one element remains, pop() and steal() race:
//   - pop() stores bottom=b-1 (claiming last slot), then reads top.
//   - steal() reads top, then reads bottom (to verify work exists).
//   On ARM/POWER these loads and stores can be reordered without fences.
//   The seq_cst fences enforce a total order:
//     If pop's fence comes first: steal sees bottom=b-1 (empty), aborts.
//     If steal's fence comes first: steal sees bottom=b, wins CAS; pop loses.
//   Either way, exactly one thread returns the element. Safe.
//   On x86 (TSO): the fences compile to MFENCE but are guaranteed correct;
//   the CAS (LOCK CMPXCHG) alone would suffice on x86 but the fences ensure
//   portability to ARM.
//
// Why can data[t%cap] be loaded relaxed in steal()?
//   steal()'s acquire load of bottom synchronizes with push()'s fence-release
//   sequence (release fence → relaxed store to bottom). This sync establishes
//   that all data stores done before push()'s fence are visible to steal()
//   after it acquires the updated bottom. Transitively, data[t%cap] (stored
//   by a preceding push()) is visible to steal() without a per-element fence.
//
// Array growth and reclamation:
//   Old arrays are kept in a trash list (freed at deque destruction). Thieves
//   may still be reading an old array mid-steal when the owner grows; since
//   all elements are copied to the new array with the same modular indices,
//   reading from either old or new array gives correct values. The trash list
//   ensures the old array's memory is not freed until the deque is destroyed.
//   A production implementation would use EBR or HP for eager reclamation.
//
// T requirements:
//   T must be trivially copyable — stored in std::atomic<T>[], which requires
//   trivial copy for atomic load/store without locks. For non-trivial tasks
//   (lambdas, std::function), store T* (a pointer to a heap-allocated task).
//   The work-stealing thread pool (step 12) uses this pattern.

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace foundation {

template <typename T>
class ChaseLevDeque {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ChaseLevDeque<T>: T must be trivially copyable for lock-free "
                  "atomic storage. Store T* for non-trivial task types.");

    // Circular buffer with power-of-2 capacity for cheap modulo via bitmasking.
    struct Array {
        const std::size_t cap;   // capacity (power of 2)
        const std::size_t mask;  // cap - 1, for index masking
        std::unique_ptr<std::atomic<T>[]> data;

        explicit Array(std::size_t c)
            : cap(c), mask(c - 1), data(new std::atomic<T>[c]) {
            assert(c > 0 && (c & (c - 1)) == 0);
        }

        // i is always non-negative at call sites; the cast makes that explicit.
        T load(int64_t i) const noexcept {
            return data[static_cast<std::size_t>(i) & mask].load(std::memory_order_relaxed);
        }
        void store(int64_t i, T v) noexcept {
            data[static_cast<std::size_t>(i) & mask].store(v, std::memory_order_relaxed);
        }
    };

public:
    explicit ChaseLevDeque(std::size_t initial_cap = 256)
        : array_(new Array(initial_cap)) {
        assert(initial_cap > 0 && (initial_cap & (initial_cap - 1)) == 0);
    }

    ~ChaseLevDeque() noexcept {
        delete array_.load(std::memory_order_relaxed);
    }

    ChaseLevDeque(const ChaseLevDeque&)            = delete;
    ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;

    // OWNER ONLY: push val onto the bottom.
    void push(T val) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);
        Array*  a = array_.load(std::memory_order_relaxed);

        if (static_cast<std::size_t>(b - t) >= a->cap)
            a = grow(a, t, b);

        a->store(b, val);
        // Release store on bottom_: all stores sequenced before this
        // (including a->store) are visible to any steal() that acquires bottom_.
        // Equivalent to the original fence(release) + relaxed-bottom pattern
        // per C++11 §29.8p2, but a single release store is directly visible
        // to TSan's acquire/release model (TSan cannot always track fence-based
        // synchronization through compound fence + relaxed-store sequences).
        bottom_.store(b + 1, std::memory_order_release);
    }

    // OWNER ONLY: pop from the bottom (LIFO).
    // Returns nullopt if empty or if a concurrent steal won the last element.
    std::optional<T> pop() {
        int64_t b  = bottom_.load(std::memory_order_relaxed) - 1;
        Array*  a  = array_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_seq_cst);

        int64_t t = top_.load(std::memory_order_relaxed);

        if (t > b) {
            // Deque is empty (or a thief claimed the last element).
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }

        T val = a->load(b);

        if (t == b) {
            // Last element: race with concurrent steal().
            // Use seq_cst CAS to pair with steal's seq_cst CAS.
            if (!top_.compare_exchange_strong(t, t + 1,
                                              std::memory_order_seq_cst,
                                              std::memory_order_relaxed)) {
                // Lost to a thief — they got the last element.
                bottom_.store(b + 1, std::memory_order_relaxed);
                return std::nullopt;
            }
            // Won the CAS. Advance bottom past the element we just claimed.
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return val;
    }

    // ANY THREAD: steal from the top (FIFO).
    // Returns nullopt if empty or if another thread won the CAS.
    std::optional<T> steal() {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t >= b) return std::nullopt;

        // array_ acquire: ensures we see the data published with the array.
        Array* a  = array_.load(std::memory_order_acquire);
        T      val = a->load(t);  // relaxed: ordered via bottom acquire (see doc)

        if (!top_.compare_exchange_strong(t, t + 1,
                                          std::memory_order_seq_cst,
                                          std::memory_order_relaxed))
            return std::nullopt;  // another stealer or pop won

        return val;
    }

    // Approximate element count. Not linearizable; use only in quiescent state
    // or for monitoring. May undercount during concurrent pushes.
    std::size_t size() const noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b > t ? static_cast<std::size_t>(b - t) : 0;
    }

    bool empty() const noexcept { return size() == 0; }

private:
    Array* grow(Array* old, int64_t t, int64_t b) {
        auto* a = new Array(old->cap * 2);
        for (int64_t i = t; i < b; ++i)
            a->store(i, old->load(i));
        trash_.push_back(std::unique_ptr<Array>(old));
        array_.store(a, std::memory_order_release);
        return a;
    }

    alignas(64) std::atomic<int64_t> top_{0};
    alignas(64) std::atomic<int64_t> bottom_{0};
    alignas(64) std::atomic<Array*>  array_;
    std::vector<std::unique_ptr<Array>> trash_;  // old arrays freed at destruction
};

} // namespace foundation
