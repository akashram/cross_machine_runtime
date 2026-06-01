#pragma once

// Lock-free Treiber stack with epoch-based memory reclamation.
//
// Compare with HazardStack (step 6):
//   - pop() opens a critical section with EpochDomain::Guard instead of
//     registering a per-node hazard pointer.
//   - Overhead per pop: one seq_cst store (enter) + one release store (exit)
//     regardless of how many nodes are examined in the CAS loop.
//   - No per-node protection registration — simpler hot path.
//   - Reclamation lag: objects are freed only after the epoch advances twice,
//     which requires ALL threads to refresh their local epoch. A thread that
//     stalls in a section blocks reclamation for everyone.
//
// No ABA guard: unlike AbaStack/HazardStack, this uses a plain pointer CAS.
// ABA cannot corrupt the stack here because:
//   - A node X is only reachable as a head pointer.
//   - For X to be pushed back (ABA), the pusher must have popped X first.
//   - Popping X requires a successful CAS that removes X from head.
//   - Any thread that loaded head = X is in a critical section (epoch guard).
//   - EBR prevents X from being freed (and its memory reused by a new node
//     with the same address) while any such critical section is active.
//   - Therefore, the ABA sequence cannot occur while any thread holds a
//     reference to X — the allocator cannot return X's address to a new
//     allocation until X is retired, and X cannot be retired until popped.
//
// So: EBR implicitly prevents ABA on the nodes accessed during a guard.
// The tagged pointer (cmpxchg16b) is not needed here.

#include "epoch_reclaim.h"

namespace foundation {

template <typename T>
class EpochStack {
public:
    explicit EpochStack(EpochDomain& domain) noexcept : domain_(domain) {}

    EpochStack(const EpochStack&)            = delete;
    EpochStack& operator=(const EpochStack&) = delete;

    ~EpochStack() noexcept {
        // Drain. All threads must have stopped using this stack first.
        while (drain_one()) {}
    }

    void push(T val) noexcept(false) {
        auto* n = new Node{std::move(val), nullptr};
        Node* old = head_.load(std::memory_order_acquire);
        do {
            n->next = old;
        } while (!head_.compare_exchange_weak(old, n,
                                              std::memory_order_release,
                                              std::memory_order_acquire));
    }

    // Pop a value. Returns false if empty.
    // Opens a critical section for the CAS loop — all node accesses inside
    // are protected by EBR. The node is retired after the section closes.
    bool pop(T& val) noexcept {
        Node* removed = nullptr;

        {
            EpochDomain::Guard guard(domain_);
            Node* old = head_.load(std::memory_order_acquire);
            while (old) {
                Node* next = old->next;
                if (head_.compare_exchange_weak(old, next,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
                    val     = std::move(old->data);
                    removed = old;
                    break;
                }
            }
        } // exit() — close critical section before retiring

        if (removed) {
            domain_.retire(removed);
            return true;
        }
        return false;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

private:
    struct Node {
        T     data;
        Node* next{nullptr};
    };

    std::atomic<Node*> head_{nullptr};
    EpochDomain&       domain_;

    // Drain without materialising T — used only by destructor.
    bool drain_one() noexcept {
        Node* removed = nullptr;
        {
            EpochDomain::Guard guard(domain_);
            Node* old = head_.load(std::memory_order_acquire);
            if (!old) return false;
            if (head_.compare_exchange_strong(old, old->next,
                                              std::memory_order_release,
                                              std::memory_order_acquire))
                removed = old;
        }
        if (removed) { domain_.retire(removed); return true; }
        return true;  // CAS lost but stack may still have items — keep draining
    }
};

} // namespace foundation
