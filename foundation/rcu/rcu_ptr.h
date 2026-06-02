#pragma once

// RcuPtr<T>: an RCU-protected, read-mostly atomic pointer.
//
// Canonical use case: a configuration object, routing table, or subscriber
// list that is read millions of times per second but replaced rarely.
//
// Protocol:
//   Reader:
//     {
//         RcuDomain::ReadGuard guard(domain);
//         T* p = rcu_ptr.get();      // acquire load — safe while guard is held
//         // ... use *p freely ...
//     }  // guard releases — writer may now free the old object
//
//   Writer:
//     T* new_obj = new T{...};       // build new version
//     rcu_ptr.store(new_obj);        // publish atomically, retire old
//     // old object is freed after the next grace period
//
// Memory model:
//   store() uses acq_rel exchange — all writes to *new_obj before store()
//   are visible to any reader whose get() observes new_obj. get() uses
//   acquire to pair with that release.
//
// Ownership: RcuPtr owns the pointed-to object. ~RcuPtr() deletes it
// directly (no retire) — callers must ensure no readers are active when
// the RcuPtr is destroyed.

#include "rcu_domain.h"

namespace foundation {

template <typename T>
class RcuPtr {
public:
    explicit RcuPtr(RcuDomain& domain, T* initial = nullptr) noexcept
        : domain_(domain), ptr_(initial) {}

    ~RcuPtr() noexcept {
        delete ptr_.load(std::memory_order_acquire);
    }

    RcuPtr(const RcuPtr&)            = delete;
    RcuPtr& operator=(const RcuPtr&) = delete;

    // Reader: load the current pointer with acquire semantics.
    // MUST be called while holding a ReadGuard. The returned pointer is valid
    // for the duration of that guard — do not use it after the guard exits.
    [[nodiscard]] T* get() const noexcept {
        return ptr_.load(std::memory_order_acquire);
    }

    // Writer: atomically publish new_val and retire the old pointer.
    // new_val must be heap-allocated; it will be freed (via domain_.retire)
    // once a grace period has elapsed. May block if the retire threshold fires.
    void store(T* new_val) noexcept {
        T* old = ptr_.exchange(new_val, std::memory_order_acq_rel);
        if (old) domain_.retire(old);
    }

private:
    RcuDomain&   domain_;
    std::atomic<T*> ptr_;
};

} // namespace foundation
