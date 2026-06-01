#pragma once

// Hazard pointer memory reclamation (header-only).
//
// Based on: Maged Michael, "Hazard Pointers: Safe Memory Reclamation for
// Lock-Free Objects", IEEE TPDS 2004.
//
// Problem: in a lock-free pop(), thread A loads head = X, reads X->next,
// then executes CAS. Between the load and the CAS, thread B can pop X,
// free it, and allocate a new node at the same address. Thread A's CAS
// may succeed with a dangling X->next — a use-after-free.
//
// Solution: before dereferencing a pointer obtained from a shared location,
// publish it as a "hazard pointer". Any thread wanting to free a pointer
// first scans all published hazard pointers. If the pointer appears there,
// it is deferred to a per-thread retire list and retried later. A pointer
// is only freed once no hazard pointer protects it.
//
// Protocol (Michael 2004, section 2.2):
//   Reader:
//     (1) hp[j] = p              (seq_cst store — advertise intent to access p)
//     (2) if (src.load(seq_cst) != p) goto (1)   (validate p still reachable)
//     (3) dereference p safely   (p cannot be freed while hp[j] = p)
//     (4) hp[j] = nullptr
//
//   Retiring thread:
//     (1) remove p from the data structure
//     (2) retire_list.push_back(p)
//     (3) if (retire_list.size() >= threshold) scan()
//
//   scan():
//     (1) collect all published hazard pointers into set H
//     (2) for p in retire_list: if p not in H → free(p), else keep
//
// Why seq_cst on slots?
//   With only acquire/release, a scan() could miss a concurrently-published
//   hazard pointer if the acquire load and the release store are on different
//   cache lines and the architecture allows reordering (e.g., ARM). Using
//   seq_cst on all slot operations places them in a single total order,
//   guaranteeing that any scan() executed after a protect() sees the
//   hazard pointer. On x86 (TSO) seq_cst compiles to the same instructions
//   as release/acquire; on ARM it emits an additional full barrier.
//
// Threshold:
//   Scan when retire_list.size() >= kScanFactor * total_hazard_slots.
//   kScanFactor = 2 gives amortized O(1) cost per retire() (each freed
//   pointer pays for at most one scan pass of the hazard pointer list).
//
// Limitation:
//   The thread_local state in tl() is shared across all HazardDomain
//   instances. Using more than one domain per thread will assert-fail.
//   This is sufficient for all Phase 1 data structures.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <vector>

namespace foundation {

// Number of hazard pointer slots per thread. K=2 is enough for all
// algorithms in this project (pop() needs at most 1; Michael-Scott
// queue needs 2 when protecting both head and next simultaneously).
inline constexpr std::size_t kHazardSlotsPerThread = 2;

// Trigger a scan when the retire list has >= kScanFactor * total_slots entries.
inline constexpr std::size_t kScanFactor = 2;

// ---------------------------------------------------------------------------
// HazardRecord: per-thread node in the global linked list.
// Records are never freed — threads acquire and release them, but the
// record object persists for the domain's lifetime. This avoids the
// bootstrapping problem of protecting the record list with hazard pointers.
// ---------------------------------------------------------------------------
struct alignas(64) HazardRecord {
    std::atomic<void*>         slots[kHazardSlotsPerThread]{};
    std::atomic<bool>          active{false};
    std::atomic<HazardRecord*> next{nullptr};
};

// ---------------------------------------------------------------------------
// RetiredPtr: a pointer scheduled for deferred deletion.
// ---------------------------------------------------------------------------
struct RetiredPtr {
    void* ptr{nullptr};
    void(*deleter)(void*) noexcept {nullptr};

    void reclaim() const noexcept { deleter(ptr); }
};

// ---------------------------------------------------------------------------
// HazardDomain
// ---------------------------------------------------------------------------
class HazardDomain {
public:
    // RAII guard: holds one hazard pointer slot for the duration of its life.
    // Usage:
    //   HazardDomain::Guard guard(domain, /*slot=*/0);
    //   T* p = guard.protect(atomic_head);
    //   // ... safely dereference p ...
    //   guard.reset();              // or let destructor clear it
    //   domain.retire(p);          // after removing from structure
    class Guard {
    public:
        explicit Guard(HazardDomain& d, std::size_t slot = 0) noexcept
            : d_(d), slot_(slot) {}
        ~Guard() noexcept { reset(); }

        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;

        // Load src and register as hazard pointer. Loops until the hazard
        // pointer and the source are consistent (Michael 2004 validate step).
        // Returns the protected pointer; safe to dereference until reset().
        template <typename T>
        [[nodiscard]] T* protect(const std::atomic<T*>& src) noexcept {
            T* p;
            do {
                p = src.load(std::memory_order_relaxed);
                d_.set_slot(slot_, static_cast<void*>(p));
                // seq_cst load: if any retiring thread's scan (also seq_cst)
                // has already passed this slot, it will see our hazard pointer
                // because seq_cst operations form a single total order.
            } while (src.load(std::memory_order_seq_cst) != p);
            return p;
        }

        // Set slot directly (no validation). Use only when p is provably
        // still reachable (e.g., you own it and just want to transfer the guard).
        void protect_raw(void* p) noexcept { d_.set_slot(slot_, p); }

        // Clear the hazard pointer.
        void reset() noexcept { d_.set_slot(slot_, nullptr); }

    private:
        HazardDomain& d_;
        std::size_t   slot_;
    };

    // -----------------------------------------------------------------------
    // retire(): schedule ptr for deferred deletion.
    // Must be called only AFTER ptr has been removed from all shared structures
    // (i.e., no new thread can obtain a reference to ptr via the data structure).
    // -----------------------------------------------------------------------
    template <typename T>
    void retire(T* ptr) noexcept {
        do_retire(static_cast<void*>(ptr),
                  [](void* p) noexcept { delete static_cast<T*>(p); });
    }

    // Force a hazard pointer scan and reclaim everything currently unprotected.
    // Called automatically when the retire list exceeds the threshold, but may
    // be called manually (e.g., in tests to verify that memory is actually freed).
    void scan() noexcept;

    // Number of objects awaiting reclamation on the current thread's retire list.
    std::size_t pending_count() const noexcept;

    // Number of HazardRecord objects registered (== number of threads that have
    // ever used this domain).
    std::size_t record_count() const noexcept {
        return record_count_.load(std::memory_order_acquire);
    }

    // -----------------------------------------------------------------------
    // Per-thread state: only the hazard pointer record.
    // The retire list is global (inside the domain) so it survives thread exit.
    // -----------------------------------------------------------------------
    struct ThreadState {
        uint64_t      domain_id{0};   // 0 = unregistered
        HazardRecord* record{nullptr};

        ~ThreadState() noexcept;
    };

private:
    ThreadState&  tl() noexcept;
    void          do_retire(void* ptr, void(*del)(void*) noexcept) noexcept;
    void          set_slot(std::size_t slot, void* p) noexcept;
    HazardRecord* get_or_create_record() noexcept;

    static inline std::atomic<uint64_t> next_id_{1};
    const uint64_t id_{next_id_.fetch_add(1, std::memory_order_relaxed)};

    alignas(64) std::atomic<HazardRecord*> head_{nullptr};
    alignas(64) std::atomic<std::size_t>  record_count_{0};

    // Global retire list. Protected by a mutex — held only during retire/scan,
    // never during hazard pointer slot operations (which are lock-free).
    mutable std::mutex       retire_mutex_;
    std::vector<RetiredPtr>  global_retire_;
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------

// tl() returns the thread-local ThreadState.
// The function-local thread_local avoids ODR issues in a header-only library.
// Limitation: all HazardDomain instances share the same thread_local, so
// each thread may only use one domain at a time.
inline HazardDomain::ThreadState& HazardDomain::tl() noexcept {
    thread_local ThreadState s;
    if (s.domain_id != id_) {
        if (s.record)
            s.record->active.store(false, std::memory_order_release);
        s.domain_id = id_;
        s.record    = get_or_create_record();
    }
    return s;
}

inline HazardDomain::ThreadState::~ThreadState() noexcept {
    // Just release the hazard pointer record. The retire list lives in the
    // domain (global_retire_) so items are not lost when this thread exits.
    if (record)
        record->active.store(false, std::memory_order_release);
}

inline HazardRecord* HazardDomain::get_or_create_record() noexcept {
    // Scan for an inactive record to reuse.
    HazardRecord* r = head_.load(std::memory_order_acquire);
    while (r) {
        bool expected = false;
        if (r->active.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel))
            return r;
        r = r->next.load(std::memory_order_acquire);
    }
    // None found — allocate and prepend.
    auto* nr = new HazardRecord;
    nr->active.store(true, std::memory_order_relaxed);
    HazardRecord* old = head_.load(std::memory_order_acquire);
    do {
        nr->next.store(old, std::memory_order_relaxed);
    } while (!head_.compare_exchange_weak(old, nr,
                                          std::memory_order_release,
                                          std::memory_order_acquire));
    record_count_.fetch_add(1, std::memory_order_relaxed);
    return nr;
}

inline void HazardDomain::set_slot(std::size_t slot, void* p) noexcept {
    assert(slot < kHazardSlotsPerThread);
    // seq_cst store: visible to all concurrent scan() seq_cst loads in total order.
    tl().record->slots[slot].store(p, std::memory_order_seq_cst);
}

inline void HazardDomain::do_retire(void* ptr, void(*del)(void*) noexcept) noexcept {
    std::size_t list_size;
    {
        std::lock_guard lk(retire_mutex_);
        global_retire_.push_back({ptr, del});
        list_size = global_retire_.size();
    }
    // Threshold: 2 * (number of records) * slots_per_thread.
    std::size_t n = record_count_.load(std::memory_order_acquire);
    std::size_t threshold = kScanFactor * n * kHazardSlotsPerThread;
    if (threshold == 0) threshold = 1;
    if (list_size >= threshold)
        scan();
}

inline void HazardDomain::scan() noexcept {
    // Take ownership of the current retire list.
    std::vector<RetiredPtr> to_process;
    {
        std::lock_guard lk(retire_mutex_);
        if (global_retire_.empty()) return;
        to_process = std::move(global_retire_);
    }

    // Collect all published hazard pointers (seq_cst).
    std::vector<void*> hazards;
    hazards.reserve(record_count_.load(std::memory_order_acquire) *
                    kHazardSlotsPerThread);
    HazardRecord* r = head_.load(std::memory_order_acquire);
    while (r) {
        for (std::size_t i = 0; i < kHazardSlotsPerThread; ++i) {
            void* p = r->slots[i].load(std::memory_order_seq_cst);
            if (p) hazards.push_back(p);
        }
        r = r->next.load(std::memory_order_acquire);
    }
    std::sort(hazards.begin(), hazards.end());

    // Free unprotected items; return survivors to the global list.
    std::vector<RetiredPtr> survivors;
    for (auto& rp : to_process) {
        if (!std::binary_search(hazards.begin(), hazards.end(), rp.ptr))
            rp.reclaim();
        else
            survivors.push_back(rp);
    }
    if (!survivors.empty()) {
        std::lock_guard lk(retire_mutex_);
        global_retire_.insert(global_retire_.end(),
                              survivors.begin(), survivors.end());
    }
}

inline std::size_t HazardDomain::pending_count() const noexcept {
    std::lock_guard lk(retire_mutex_);
    return global_retire_.size();
}

} // namespace foundation
