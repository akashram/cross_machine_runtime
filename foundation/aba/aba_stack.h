#pragma once

#include <atomic>
#include <cstdint>

namespace foundation {

// AbaStack: a lock-free Treiber stack that eliminates ABA via tagged pointers.
//
// Fix: replace the single-word CAS on a raw pointer with a double-word CAS
// (cmpxchg16b on x86-64) on a {ptr, tag} pair. The tag is a monotonic counter
// incremented on every push and pop. Even if the pointer value cycles back to A,
// the tag differs, so the CAS correctly fails.
//
//   Initial: head = {&A, tag=2}
//   Thread 2: pop A → {&B, 3}, pop B → {nullptr, 4}, push A → {&A, 5}
//   Thread 1: CAS({&A, 2}, {&B, 2}) → FAILS  (current tag is 5, not 2)
//   Thread 1: retries with {&A, 5}, CAS({&A, 5}, {nullptr, 6}) → SUCCEEDS
//   Correct outcome: head = nullptr, both pops got the right nodes.
//
// TaggedPtr is 16 bytes: 8 for the pointer, 8 for the tag.
// On x86-64, std::atomic<16-byte-struct> uses cmpxchg16b, which is lock-free
// when the struct is 16-byte aligned. The static_assert below enforces this.
//
// Why not pack the tag into spare pointer bits?
//   x86-64 canonical addresses use 48 bits (57 with LA57). We could pack a
//   16-bit tag into the upper bits. A 16-bit tag wraps at 65536 ops inside the
//   ABA window — sufficient for nearly all workloads, but the 16-byte approach
//   gives a full 64-bit tag with no masking overhead on every load, and the
//   intent is clearer.
//
// Memory reclamation note:
//   Tagged pointers prevent ABA on the head CAS, but they do NOT solve
//   use-after-free: a thread holding old.ptr->next holds no reference
//   preventing deletion of old.ptr. pop() therefore intentionally leaks the
//   node. Safe reclamation is deferred to step 6 (hazard pointers) and
//   step 7 (epoch-based reclamation).
//
// head_ is public to allow the ABA scenario demonstration test.

template <typename T>
class AbaStack {
public:
    struct Node {
        T     data;
        Node* next{nullptr};
    };

    struct TaggedPtr {
        Node*    ptr{nullptr};
        uint64_t tag{0};
        bool operator==(const TaggedPtr&) const = default;
    };

    static_assert(sizeof(TaggedPtr) == 16,
        "TaggedPtr must be 16 bytes for cmpxchg16b");
    static_assert(std::atomic<TaggedPtr>::is_always_lock_free,
        "AbaStack requires a 16-byte lock-free CAS (cmpxchg16b on x86-64). "
        "Ensure the target supports cmpxchg16b and TaggedPtr is 16-byte aligned.");

    alignas(16) std::atomic<TaggedPtr> head_{TaggedPtr{}};

    void push(Node* n) noexcept {
        TaggedPtr old = head_.load(std::memory_order_acquire);
        TaggedPtr next;
        do {
            n->next = old.ptr;
            next    = TaggedPtr{n, old.tag + 1};
        } while (!head_.compare_exchange_weak(old, next,
                                              std::memory_order_release,
                                              std::memory_order_acquire));
    }

    // Returns the popped Node* (caller owns it), or nullptr if empty.
    // The caller must not free the node while other threads may still hold
    // a pointer to it — see the memory reclamation note above.
    Node* pop() noexcept {
        TaggedPtr old = head_.load(std::memory_order_acquire);
        TaggedPtr next;
        while (old.ptr) {
            next = TaggedPtr{old.ptr->next, old.tag + 1};
            if (head_.compare_exchange_weak(old, next,
                                            std::memory_order_release,
                                            std::memory_order_acquire))
                return old.ptr;
        }
        return nullptr;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire).ptr == nullptr;
    }
};

} // namespace foundation
