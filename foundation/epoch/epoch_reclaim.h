#pragma once

// Epoch-based memory reclamation (EBR), header-only.
//
// EBR is an alternative to hazard pointers (step 6) with a different
// trade-off profile:
//
//   Read side:  enter() = one atomic store + one atomic load.
//               No per-pointer registration. The whole critical section is
//               protected — you can dereference as many pointers as you like
//               without extra overhead.
//
//   Reclamation: try_advance() scans all thread records to check if a new
//               epoch can begin. Cost is O(T) where T is the thread count.
//               When the epoch advances, objects two epochs old are freed.
//
// Algorithm (Fraser 2004, simplified):
//
//   Global epoch G, starting at 0. Three retire slots indexed [0, 1, 2],
//   cycling as G mod 3.
//
//   enter():  read G, set local_epoch = G, set active = true (seq_cst)
//   exit():   set active = false
//
//   retire(p): append p to global_retire[G mod 3]
//              then call try_advance()
//
//   try_advance():
//     read G_current
//     for each active thread: if local_epoch != G_current → return (not ready)
//     CAS(G, G_current, G_current + 1)   ← only one winner
//     if CAS won: reclaim global_retire[(G_current - 1) mod 3]
//
// Why is global_retire[(G-1) mod 3] safe to reclaim when advancing to G+1?
//   Objects in that slot were retired when the global epoch was (G-1).
//   Any thread that obtained a pointer to them was in a critical section
//   during epoch (G-1) and thus had local_epoch = (G-1). For the current
//   advance check (all active threads have local_epoch = G) to pass, every
//   such thread must have since exited its (G-1) section and re-entered
//   with local_epoch = G (or become inactive). Those threads no longer hold
//   the old pointers. Safe to free.
//
// EBR vs. hazard pointers — when to use each:
//
//   Choose EBR when:
//     - Operations access many pointers per critical section (linked list
//       traversal, tree walks). Hazard pointers require registering each
//       pointer individually; EBR protects all of them with one store.
//     - Short, bounded critical sections. Long sections delay reclamation
//       globally — any thread stalled in a section holds back the epoch for
//       everyone.
//
//   Choose hazard pointers when:
//     - Strict reclamation latency is required. A stalled thread in EBR
//       blocks ALL reclamation; hazard pointers only block pointers that
//       thread is actively protecting.
//     - Threads may sleep or block for extended periods inside the data
//       structure. EBR would stall the entire reclaim pipeline.
//
// Implementation notes:
//   - Retire lists are global (shared, mutex-protected). The mutex is never
//     held during the critical section (enter/exit) — only during retire()
//     and the reclamation step of try_advance(). A lock-free list could
//     replace the mutex, but adds complexity; this is sufficient for Phase 1.
//   - Thread records are allocated lazily, never freed, and reused across
//     threads (same pattern as HazardDomain).
//   - Thread-local limitation: one EpochDomain per thread (asserted).

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace foundation {

// Number of epoch slots. 3 is the minimum for the 2-epoch-lag safety argument.
inline constexpr uint64_t kEBREpochs = 3;

struct RetiredPtrE {
    void* ptr{nullptr};
    void(*deleter)(void*) noexcept {nullptr};
    void reclaim() const noexcept { deleter(ptr); }
};

// ---------------------------------------------------------------------------
// EpochDomain
// ---------------------------------------------------------------------------
class EpochDomain {
public:
    // RAII critical-section guard.
    // While alive: this thread's references are protected from reclamation.
    // Must NOT be held across blocking calls or sleeps.
    class Guard {
    public:
        explicit Guard(EpochDomain& d) noexcept : d_(d) { d_.enter(); }
        ~Guard() noexcept { d_.exit(); }
        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
    private:
        EpochDomain& d_;
    };

    // Schedule ptr for deferred deletion.
    // Must be called AFTER ptr has been removed from all shared structures.
    // May be called inside or outside a Guard (both are correct).
    template <typename T>
    void retire(T* ptr) noexcept {
        do_retire(static_cast<void*>(ptr),
                  [](void* p) noexcept { delete static_cast<T*>(p); });
    }

    // Attempt to advance the global epoch and reclaim old retire lists.
    // Called automatically by retire(); expose for tests.
    void try_advance() noexcept;

    // Total bytes of objects awaiting reclamation (count, not bytes).
    std::size_t pending_count() noexcept;

    // Current global epoch value (for introspection / testing).
    uint64_t epoch() const noexcept {
        return global_epoch_.load(std::memory_order_acquire);
    }

    // Number of threads ever registered in this domain.
    std::size_t thread_count() const noexcept {
        return thread_count_.load(std::memory_order_acquire);
    }

    // -----------------------------------------------------------------------
    // Internal per-thread state
    // -----------------------------------------------------------------------
    struct alignas(64) ThreadRecord {
        std::atomic<uint64_t>    local_epoch{0};
        std::atomic<bool>         active{false};
        std::atomic<ThreadRecord*> next{nullptr};
    };

    struct ThreadState {
        uint64_t      domain_id{0};   // 0 = unregistered
        ThreadRecord* record{nullptr};
        EpochDomain*  domain{nullptr};
        ~ThreadState() noexcept;
    };

private:
    void enter() noexcept;
    void exit()  noexcept;
    void do_retire(void* ptr, void(*del)(void*) noexcept) noexcept;

    ThreadState&  tl() noexcept;
    ThreadRecord* get_or_create_record() noexcept;

    // Unique ID per domain instance — avoids false positives when two domains
    // are constructed at the same stack address in the same thread.
    static inline std::atomic<uint64_t> next_id_{1};
    const uint64_t id_{next_id_.fetch_add(1, std::memory_order_relaxed)};

    alignas(64) std::atomic<uint64_t>    global_epoch_{0};
    alignas(64) std::atomic<ThreadRecord*> head_{nullptr};
    alignas(64) std::atomic<std::size_t> thread_count_{0};

    // Global retire lists. Mutex ensures safe concurrent push from multiple
    // threads, and safe move-out during try_advance(). The mutex is never
    // held during enter() / exit() — only at retire/reclaim time.
    std::mutex               retire_mutex_[kEBREpochs];
    std::vector<RetiredPtrE>  global_retire_[kEBREpochs];
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------

inline EpochDomain::ThreadState& EpochDomain::tl() noexcept {
    thread_local ThreadState s;
    if (s.domain_id != id_) {
        // Either first use or a new domain (different ID). Release any old record.
        if (s.record)
            s.record->active.store(false, std::memory_order_release);
        s.domain_id = id_;
        s.domain    = this;
        s.record    = get_or_create_record();
    }
    return s;
}

inline EpochDomain::ThreadState::~ThreadState() noexcept {
    if (record)
        record->active.store(false, std::memory_order_release);
}

inline EpochDomain::ThreadRecord* EpochDomain::get_or_create_record() noexcept {
    ThreadRecord* r = head_.load(std::memory_order_acquire);
    while (r) {
        bool expected = false;
        if (r->active.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel))
            return r;
        r = r->next.load(std::memory_order_acquire);
    }
    auto* nr = new ThreadRecord;
    nr->active.store(true, std::memory_order_relaxed);
    ThreadRecord* old = head_.load(std::memory_order_acquire);
    do {
        nr->next.store(old, std::memory_order_relaxed);
    } while (!head_.compare_exchange_weak(old, nr,
                                          std::memory_order_release,
                                          std::memory_order_acquire));
    thread_count_.fetch_add(1, std::memory_order_relaxed);
    return nr;
}

inline void EpochDomain::enter() noexcept {
    ThreadRecord& r = *tl().record;
    // Re-read global epoch and cache locally.
    uint64_t g = global_epoch_.load(std::memory_order_acquire);
    r.local_epoch.store(g, std::memory_order_relaxed);
    // seq_cst: ensures any concurrent try_advance() that reads active=true
    // will also observe local_epoch = g (not an older value).
    r.active.store(true, std::memory_order_seq_cst);
}

inline void EpochDomain::exit() noexcept {
    tl().record->active.store(false, std::memory_order_release);
}

inline void EpochDomain::do_retire(void* ptr, void(*del)(void*) noexcept) noexcept {
    uint64_t g = global_epoch_.load(std::memory_order_acquire);
    {
        std::lock_guard lk(retire_mutex_[g % kEBREpochs]);
        global_retire_[g % kEBREpochs].push_back({ptr, del});
    }
    try_advance();
}

inline void EpochDomain::try_advance() noexcept {
    uint64_t G = global_epoch_.load(std::memory_order_acquire);

    // Check every active thread has observed the current epoch.
    // seq_cst loads synchronize with the seq_cst store in enter(), guaranteeing
    // we see the most recent local_epoch written by each active thread.
    ThreadRecord* r = head_.load(std::memory_order_acquire);
    while (r) {
        if (r->active.load(std::memory_order_seq_cst)) {
            if (r->local_epoch.load(std::memory_order_acquire) != G)
                return;  // at least one thread is behind — not safe to advance
        }
        r = r->next.load(std::memory_order_acquire);
    }

    // Race to advance the epoch. Only one thread wins.
    if (!global_epoch_.compare_exchange_strong(G, G + 1,
                                               std::memory_order_acq_rel))
        return;

    // We advanced from G to G+1. Reclaim slot (G-1) mod 3.
    // Objects there were retired at least 2 epoch-advances ago — safe.
    const uint64_t old_slot = (G - 1 + kEBREpochs) % kEBREpochs;
    std::vector<RetiredPtrE> to_reclaim;
    {
        std::lock_guard lk(retire_mutex_[old_slot]);
        to_reclaim = std::move(global_retire_[old_slot]);
    }
    for (auto& rp : to_reclaim) rp.reclaim();
}

inline std::size_t EpochDomain::pending_count() noexcept {
    std::size_t total = 0;
    for (uint64_t i = 0; i < kEBREpochs; ++i) {
        std::lock_guard lk(retire_mutex_[i]);
        total += global_retire_[i].size();
    }
    return total;
}

} // namespace foundation
