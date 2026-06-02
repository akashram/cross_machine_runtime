#pragma once

// FreeList<T>: fixed-capacity lock-free object pool, header-only.
//
// A pool of N pre-allocated T-sized slots managed as a lock-free Treiber
// stack. acquire() pops a free slot in O(1) amortized; release() pushes it
// back. No heap allocation occurs after construction.
//
// ABA prevention — why tagged pointers and not hazard pointers:
//
//   The ABA problem in a recycle pool:
//     Thread A: reads head = slot X, reads next[X] = slot Y
//     Thread B: acquire()s X, uses it, release()s X back — head = X again,
//               next[X] overwritten to whatever was at head during push
//     Thread A: CAS(head, X, Y) succeeds with stale Y — Y may be in use
//
//   Hazard pointers prevent USE-AFTER-FREE (they defer delete until no thread
//   holds a HP to that address). But pool release() does NOT call delete; it
//   just rewrites next[X] and pushes X back. HP cannot observe this overwrite
//   and therefore cannot block it. HP does NOT prevent ABA in a recycle pool.
//
//   HP IS the right tool when pool slots can be permanently freed (pool
//   shrink or drain while threads are actively acquiring). Step 14 (arena
//   allocator) will combine FreeList with an HP-protected drain path.
//
//   A monotone tag solves ABA: the head carries a 32-bit counter that
//   increments on every CAS. If slot X returns to the top, its tag differs
//   from thread A's snapshot, so A's CAS fails and it re-reads head fresh.
//
// Layout — why separate storage and next arrays:
//
//   A naive approach stores the next pointer IN the slot via a union:
//       union Slot { alignas(T) byte storage[sizeof(T)]; Slot* next; };
//   This lets acquire() read slot->next and another thread write *p to
//   slot->storage concurrently. The union makes them alias the same address,
//   which is a C++ memory model data race even though the CAS logic prevents
//   any incorrect observable outcome. ThreadSanitizer correctly flags it.
//
//   Solution: store next links in a SEPARATE atomic array. storage_[i] holds
//   the T-sized user data; next_[i] holds the index of the next free slot.
//   These arrays never overlap, so concurrent accesses through acquire() /
//   release() / user code are all on distinct atomic objects. TSan clean.
//
// Head encoding:
//   head_ is a single uint64_t packing {idx: uint32, tag: uint32}.
//   32-bit index → max 4G slots. 32-bit tag → wraps at 4G pops per slot;
//   at 100M pops/sec a slot wraps in ~43s — acceptable for all Phase 1
//   workloads (long-running servers would use a 64-bit tag via cmpxchg16b).
//   kNone (UINT32_MAX) is the sentinel for "list empty."
//
// Memory ordering:
//   acquire(): release CAS ensures the caller's subsequent writes to the slot
//     are ordered after the slot is removed from the free list.
//   release(): release CAS ensures the caller's preceding writes to the slot
//     are visible to the next thread that acquire()s it (via the acquire CAS).
//   next_[i] accesses: acquire load when reading the next slot in the chain;
//     release store when writing the chain during release(). This establishes
//     the happens-before needed for the memory ordering argument above.

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace foundation {

template <typename T>
class FreeList {
    static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

    // head_ packs (slot index, ABA tag) into 64 bits for a standard CAS.
    struct Head { uint32_t idx; uint32_t tag; };
    static_assert(sizeof(Head) == 8);

    static uint64_t pack(Head h) noexcept {
        uint64_t v; __builtin_memcpy(&v, &h, 8); return v;
    }
    static Head unpack(uint64_t v) noexcept {
        Head h; __builtin_memcpy(&h, &v, 8); return h;
    }

public:
    // Pre-allocate `capacity` slots and link them into the free stack.
    explicit FreeList(std::size_t capacity) : capacity_(capacity) {
        assert(capacity > 0);
        assert(capacity < kNone);  // kNone is the sentinel

        // storage_: raw T-sized regions, aligned to T.
        // Allocate as bytes then interpret as T*. The per-slot alignment
        // requirement is met when alignof(T) divides sizeof(T), which is
        // guaranteed by the C++ object model.
        storage_ = std::make_unique<std::byte[]>(capacity * sizeof(T));
        next_    = std::make_unique<std::atomic<uint32_t>[]>(capacity);

        for (uint32_t i = 0; i + 1 < static_cast<uint32_t>(capacity); ++i)
            next_[i].store(i + 1, std::memory_order_relaxed);
        next_[capacity - 1].store(kNone, std::memory_order_relaxed);

        head_.store(pack({0, 0}), std::memory_order_relaxed);
    }

    FreeList(const FreeList&)            = delete;
    FreeList& operator=(const FreeList&) = delete;

    // Pop a free slot and return a pointer to its T-sized raw storage.
    // Returns nullptr if the pool is exhausted.
    // Caller should construct T in-place (placement new) before use.
    [[nodiscard]] T* acquire() noexcept {
        uint64_t old_raw = head_.load(std::memory_order_acquire);
        while (true) {
            Head old = unpack(old_raw);
            if (old.idx == kNone) return nullptr;

            uint32_t nxt = next_[old.idx].load(std::memory_order_acquire);
            uint64_t new_raw = pack({nxt, old.tag + 1});

            if (head_.compare_exchange_weak(old_raw, new_raw,
                                            std::memory_order_release,
                                            std::memory_order_acquire))
                return slot_ptr(old.idx);
            // old_raw updated to current head — loop retries
        }
    }

    // Return a previously acquired slot to the free stack.
    // p must be a pointer obtained from this pool's acquire().
    void release(T* p) noexcept {
        assert(p != nullptr);
        uint32_t idx = slot_idx(p);
        uint64_t old_raw = head_.load(std::memory_order_acquire);
        while (true) {
            Head old = unpack(old_raw);
            next_[idx].store(old.idx, std::memory_order_release);
            uint64_t new_raw = pack({idx, old.tag + 1});
            if (head_.compare_exchange_weak(old_raw, new_raw,
                                            std::memory_order_release,
                                            std::memory_order_acquire))
                return;
        }
    }

    // Count free slots by traversing the stack. Only valid in quiescent state.
    std::size_t available() const noexcept {
        std::size_t count = 0;
        uint32_t idx = unpack(head_.load(std::memory_order_acquire)).idx;
        while (idx != kNone) {
            ++count;
            idx = next_[idx].load(std::memory_order_acquire);
        }
        return count;
    }

    std::size_t capacity() const noexcept { return capacity_; }

private:
    T* slot_ptr(uint32_t idx) noexcept {
        return reinterpret_cast<T*>(storage_.get() + idx * sizeof(T));
    }
    uint32_t slot_idx(const T* p) const noexcept {
        auto diff = reinterpret_cast<const std::byte*>(p) - storage_.get();
        return static_cast<uint32_t>(diff / static_cast<std::ptrdiff_t>(sizeof(T)));
    }

    alignas(8) std::atomic<uint64_t>          head_{pack({kNone, 0})};
    std::size_t                               capacity_{0};
    std::unique_ptr<std::byte[]>              storage_;  // T-sized user data
    std::unique_ptr<std::atomic<uint32_t>[]>  next_;     // free list links
};

} // namespace foundation
