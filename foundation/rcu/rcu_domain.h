#pragma once

// Read-Copy-Update (RCU) — userspace implementation, header-only.
//
// RCU is built around one key asymmetry:
//
//   Read side: near-zero overhead. Two atomic increments per critical section
//              regardless of how many objects are accessed inside. No CAS,
//              no per-pointer registration, no shared epoch to check.
//
//   Write side: "copy + modify + publish + wait for grace period + free."
//               Writers pay the cost readers avoid.
//
// This makes RCU ideal for read-mostly workloads: config objects, routing
// tables, subscription lists. A structure read millions of times per second
// but written rarely (once per second) belongs behind RCuPtr<T>.
//
// The fundamental guarantee:
//   After synchronize() returns, no thread that entered a ReadGuard before
//   synchronize() was called is still inside that guard. It is safe to free
//   any data removed from shared state before calling synchronize().
//
// Algorithm — per-thread counter (based on URCU, Desnoyers et al. 2012):
//
//   Each thread owns a 64-bit counter `ctr`:
//     even value → idle (not in a read-side critical section)
//     odd  value → active (inside a read-side critical section)
//
//   read_lock():   ctr.fetch_add(1, seq_cst)  [even → odd]
//   read_unlock(): ctr.fetch_add(1, seq_cst)  [odd  → even]
//
//   synchronize():
//     1. seq_cst fence — ensures the writer's preceding stores (the pointer
//        assignment) are globally visible before we scan readers.
//     2. For each thread record: snapshot its ctr value with seq_cst load.
//     3. For each record where snapshot is odd (reader was active):
//        spin-wait until ctr != snapshot.
//        Interpretation: the reader either exited the section (ctr went even)
//        or exited and re-entered a new section (ctr is a larger odd value).
//        Either way, the original critical section is over.
//     4. seq_cst fence — ensures all stores made by completing readers are
//        visible to the writer before it frees the old data.
//
// Why is step 3 correct?
//   "ctr != snapshot" is sufficient because:
//   - If ctr went from odd→even: reader exited. Safe.
//   - If ctr went from odd→even→odd (re-entered): the reader obtained the
//     pointer in a NEW critical section that started after step 1, so they
//     loaded from the CURRENT shared structure — not the old, removed value.
//   - Counter wraparound (after 2^63 lock/unlock cycles) is unreachable in
//     practice, so we treat uint64_t as unbounded.
//
// What about readers that start AFTER the step-1 fence?
//   They see the new pointer (the release store in rcu_assign_pointer is
//   ordered before our fence by the seq_cst total order). They cannot hold
//   a reference to the old data. We don't need to wait for them.
//
// RCU vs. epoch-based reclamation (EBR, step 7):
//
//   Both protect all pointers inside one critical section without per-pointer
//   registration. The key structural difference is WHERE shared state lives:
//
//     EBR:  readers check a GLOBAL epoch counter on section entry, then
//           writers wait for all threads to acknowledge the new epoch. A
//           stalled reader blocks ALL reclamation globally (epoch can't advance).
//
//     RCU:  readers only touch their OWN per-thread counter. Writers scan
//           the set of per-thread counters during synchronize(). A stalled
//           reader blocks only the synchronize() waiting on it, not others.
//
//   Choose RCU when:
//     - Read latency is the bottleneck and synchronize() is called rarely.
//     - Readers may be long-running (they won't stall global reclamation).
//     - The data structure is truly read-mostly (config, routing tables).
//
//   Choose EBR when:
//     - Writes are frequent and you want lower reclamation latency.
//     - Critical sections are short and bounded.
//     - You want cheaper per-section entry (one seq_cst store vs. two for RCU).
//
// Memory ordering note — why seq_cst on both read_lock and read_unlock?
//   read_lock's seq_cst fetch_add ensures: the ctr increment is visible to a
//   concurrent synchronize()'s seq_cst scan. Without this total order, an ARM
//   processor could allow the scan to execute before observing the reader's
//   increment, causing synchronize() to miss an active reader.
//   read_unlock's seq_cst ensures: the synchronize() waiting on ctr != snap
//   will see the unlock after it observes all stores the reader made inside
//   the section (otherwise the writer might free data that the reader's stores
//   still reference in a write buffer).
//   On x86 (TSO): fetch_add(seq_cst) compiles to LOCK XADD — same code as
//   fetch_add(acq_rel). The seq_cst designation matters on ARM (DMB SY).

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace foundation {

inline constexpr std::size_t kRcuReclaimThreshold = 64;

struct RetiredPtrR {
    void* ptr{nullptr};
    void(*deleter)(void*) noexcept {nullptr};
    void reclaim() const noexcept { deleter(ptr); }
};

// ---------------------------------------------------------------------------
// RcuDomain
// ---------------------------------------------------------------------------
class RcuDomain {
public:
    // RAII read-side critical section.
    // Ctor: fetch_add(1, seq_cst) — marks thread as active (even → odd).
    // Dtor: fetch_add(1, seq_cst) — marks thread as idle (odd → even).
    // Must NOT be nested on the same thread.
    // Must NOT be held across blocking syscalls that outlast synchronize().
    class ReadGuard {
    public:
        explicit ReadGuard(RcuDomain& d) noexcept : d_(d) { d_.read_lock(); }
        ~ReadGuard() noexcept { d_.read_unlock(); }
        ReadGuard(const ReadGuard&)            = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;
    private:
        RcuDomain& d_;
    };

    // Block until a full grace period has elapsed.
    // Guarantees: all readers that entered a ReadGuard before this call
    // will have exited before this returns.
    void synchronize() noexcept;

    // Schedule ptr for deferred deletion after the next grace period.
    // Accumulates up to kRcuReclaimThreshold items, then triggers a batch
    // synchronize() + free. Called after removing ptr from shared state.
    template <typename T>
    void retire(T* ptr) noexcept {
        do_retire(static_cast<void*>(ptr),
                  [](void* p) noexcept { delete static_cast<T*>(p); });
    }

    // Synchronize and free all currently pending retired objects.
    // Use in tests to verify that memory is actually reclaimed.
    void reclaim_all() noexcept;

    std::size_t pending_count() noexcept {
        std::lock_guard lk(retire_mutex_);
        return pending_.size();
    }

    std::size_t thread_count() const noexcept {
        return thread_count_.load(std::memory_order_acquire);
    }

    // -----------------------------------------------------------------------
    // Internal per-thread state
    // -----------------------------------------------------------------------
    struct alignas(64) ThreadRecord {
        std::atomic<uint64_t>      ctr{0};       // even=idle, odd=in-read-section
        std::atomic<bool>          active{false}; // true while owning thread is alive
        std::atomic<ThreadRecord*> next{nullptr};
    };

    struct ThreadState {
        uint64_t      domain_id{0};
        ThreadRecord* record{nullptr};
        RcuDomain*    domain{nullptr};
        ~ThreadState() noexcept;
    };

private:
    void          read_lock()   noexcept;
    void          read_unlock() noexcept;
    void          do_retire(void* ptr, void(*del)(void*) noexcept) noexcept;
    ThreadState&  tl() noexcept;
    ThreadRecord* get_or_create_record() noexcept;

    static inline std::atomic<uint64_t> next_id_{1};
    const uint64_t id_{next_id_.fetch_add(1, std::memory_order_relaxed)};

    alignas(64) std::atomic<ThreadRecord*> head_{nullptr};
    alignas(64) std::atomic<std::size_t>   thread_count_{0};

    std::mutex               retire_mutex_;
    std::vector<RetiredPtrR> pending_;
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------

inline RcuDomain::ThreadState& RcuDomain::tl() noexcept {
    thread_local ThreadState s;
    if (s.domain_id != id_) {
        if (s.record)
            s.record->active.store(false, std::memory_order_release);
        s.domain_id = id_;
        s.domain    = this;
        s.record    = get_or_create_record();
    }
    return s;
}

inline RcuDomain::ThreadState::~ThreadState() noexcept {
    if (record)
        record->active.store(false, std::memory_order_release);
}

inline RcuDomain::ThreadRecord* RcuDomain::get_or_create_record() noexcept {
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

inline void RcuDomain::read_lock() noexcept {
    tl().record->ctr.fetch_add(1, std::memory_order_seq_cst);
}

inline void RcuDomain::read_unlock() noexcept {
    tl().record->ctr.fetch_add(1, std::memory_order_seq_cst);
}

inline void RcuDomain::synchronize() noexcept {
    // Fence 1: the writer's preceding pointer assignment (release store) is
    // now globally visible. Any reader starting AFTER this fence sees the new
    // value, not the old one, so we don't need to wait for them.
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Snapshot each registered thread's ctr. We only need to wait for threads
    // that are actively in a read section at this moment (odd ctr).
    struct Entry { ThreadRecord* rec; uint64_t snap; };
    std::vector<Entry> waitlist;

    ThreadRecord* r = head_.load(std::memory_order_acquire);
    while (r) {
        if (r->active.load(std::memory_order_acquire)) {
            uint64_t c = r->ctr.load(std::memory_order_seq_cst);
            if (c & 1)
                waitlist.push_back({r, c});
        }
        r = r->next.load(std::memory_order_acquire);
    }

    // For each active reader, wait until their counter changes.
    // ctr != snap means they exited the section (or exited and re-entered a
    // new one). Either way the original critical section is done.
    for (auto& e : waitlist) {
        while (e.rec->ctr.load(std::memory_order_seq_cst) == e.snap)
            std::this_thread::yield();
    }

    // Fence 2: all stores made by the now-completed readers are visible to us.
    // We can safely free the old data after this point.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void RcuDomain::do_retire(void* ptr, void(*del)(void*) noexcept) noexcept {
    std::size_t sz;
    {
        std::lock_guard lk(retire_mutex_);
        pending_.push_back({ptr, del});
        sz = pending_.size();
    }
    if (sz >= kRcuReclaimThreshold)
        reclaim_all();
}

inline void RcuDomain::reclaim_all() noexcept {
    synchronize();
    std::vector<RetiredPtrR> to_free;
    {
        std::lock_guard lk(retire_mutex_);
        to_free = std::move(pending_);
    }
    for (auto& rp : to_free) rp.reclaim();
}

} // namespace foundation
