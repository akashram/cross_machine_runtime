#pragma once

// Lock-free Treiber stack with hazard pointer memory reclamation.
//
// This is the production-safe version of AbaStack from step 5:
//   - push() allocates a Node with new (no node pool required)
//   - pop()  uses a hazard pointer guard to protect the node before reading
//            ->next, then retires the node after a successful CAS
//   - Memory is actually freed — no intentional leak
//
// Correctness of the pop() hazard pointer sequence:
//
//   (1) guard.protect(head_)
//       Loops loading head_ into hp[0] until head_ == hp[0] (validate).
//       After this, hp[0] = nodeX is visible to any concurrent scan() via
//       the seq_cst total order, so nodeX will not be freed while hp[0] = nodeX.
//
//   (2) head_.load() != old → retry
//       If head_ changed since we loaded it (e.g., another thread popped
//       nodeX), we restart protect() — the old hazard value is irrelevant.
//
//   (3) CAS to remove nodeX
//       Whoever's CAS wins owns nodeX exclusively. The loser retries from (1).
//
//   (4) guard.reset()
//       We own nodeX; no other thread can retire it (only we can, since
//       only the CAS winner removes it from the structure). Safe to clear.
//
//   (5) domain_.retire(nodeX)
//       nodeX is off the structure. It will be freed once no hazard pointer
//       points to it. Any concurrent reader that set hp[j] = nodeX before
//       our CAS must have validated nodeX is still in the structure; after
//       our CAS, their validate will fail and they'll retry, so they'll clear
//       hp[j] = nodeX. The next scan() will free nodeX.
//
// ABA prevention: same tagged pointer trick as AbaStack — the 16-byte
// {ptr, tag} CAS prevents a stale CAS from succeeding even if the same
// node address reappears as head after being pushed back.

#include "hazard_ptr.h"
#include <cstdint>
#include <optional>

namespace foundation {

template <typename T>
class HazardStack {
public:
    explicit HazardStack(HazardDomain& domain) noexcept : domain_(domain) {}

    HazardStack(const HazardStack&)            = delete;
    HazardStack& operator=(const HazardStack&) = delete;

    ~HazardStack() noexcept {
        // Drain the stack without materialising T values.
        // All threads must have stopped using this stack before its destructor runs.
        while (drain_one()) {}
    }

    // Push a value. Allocates a Node on the heap.
    void push(T val) noexcept(false) {
        auto* n = new Node{std::move(val), nullptr};
        TaggedPtr old = head_.load(std::memory_order_acquire);
        TaggedPtr next;
        do {
            n->next = old.ptr;
            next    = TaggedPtr{n, old.tag + 1};
        } while (!head_.compare_exchange_weak(old, next,
                                              std::memory_order_release,
                                              std::memory_order_acquire));
    }

    // Pop a value. Returns false if empty.
    // Uses a hazard pointer to protect the head node during the read of ->next.
    bool pop(T& val) noexcept {
        HazardDomain::Guard guard(domain_, 0);
        TaggedPtr old;

        while (true) {
            // Step 1: protect the current head.
            Node* head_ptr;
            do {
                old      = head_.load(std::memory_order_relaxed);
                head_ptr = old.ptr;
                guard.protect_raw(static_cast<void*>(head_ptr));
                // Validate: re-read head_ seq_cst to ensure our hazard is
                // consistent with what we'll dereference.
            } while (head_.load(std::memory_order_seq_cst).ptr != head_ptr);

            if (!head_ptr) return false;  // empty

            // Step 2: safe to read head_ptr->next (hazard pointer protects it).
            TaggedPtr next = TaggedPtr{head_ptr->next, old.tag + 1};

            // Step 3: try to remove head_ptr from the stack.
            if (head_.compare_exchange_weak(old, next,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
                // Step 4: we own head_ptr — clear hazard before retiring.
                val = std::move(head_ptr->data);
                guard.reset();
                // Step 5: retire; will be freed once no hazard points to it.
                domain_.retire(head_ptr);
                return true;
            }
            // CAS lost — another thread popped head_ptr. Retry with new head.
        }
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire).ptr == nullptr;
    }

private:
    struct Node {
        T     data;
        Node* next{nullptr};
    };

    struct TaggedPtr {
        Node*    ptr{nullptr};
        uint64_t tag{0};
    };
    static_assert(sizeof(TaggedPtr) == 16);
    static_assert(std::atomic<TaggedPtr>::is_always_lock_free,
        "HazardStack requires 16-byte lock-free CAS (cmpxchg16b on x86-64).");

    alignas(16) std::atomic<TaggedPtr> head_{TaggedPtr{}};
    HazardDomain&                      domain_;

    // Used only by the destructor: remove and retire the head node without
    // moving its T value. Returns true while the stack is non-empty.
    bool drain_one() noexcept {
        HazardDomain::Guard guard(domain_, 0);
        TaggedPtr old;
        while (true) {
            Node* head_ptr;
            do {
                old      = head_.load(std::memory_order_relaxed);
                head_ptr = old.ptr;
                guard.protect_raw(static_cast<void*>(head_ptr));
            } while (head_.load(std::memory_order_seq_cst).ptr != head_ptr);
            if (!head_ptr) return false;
            TaggedPtr next = TaggedPtr{head_ptr->next, old.tag + 1};
            if (head_.compare_exchange_weak(old, next,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
                guard.reset();
                domain_.retire(head_ptr);
                return true;
            }
        }
    }
};

} // namespace foundation
