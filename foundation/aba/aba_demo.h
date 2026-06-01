#pragma once

#include <atomic>

namespace foundation {

// BuggyStack: a lock-free Treiber stack with a documented ABA vulnerability.
//
// DO NOT USE IN PRODUCTION. This is an educational artifact.
//
// The ABA problem in pop():
//
//   Suppose the stack is A -> B -> nullptr.
//
//   Step 1. Thread 1 loads head = &A. It saves next = A->next = &B.
//           Thread 1 is preempted before executing the CAS.
//
//   Step 2. Thread 2 runs:
//             pop() → returns &A  (head = &B)
//             pop() → returns &B  (head = nullptr)
//             push(&A)            (head = &A, A->next = nullptr)
//
//   Step 3. Thread 1 resumes. It executes:
//             CAS(head, &A, &B)
//           head is still &A, so the CAS *succeeds*.
//           head is now &B — but &B was already freed by Thread 2.
//           Thread 2 has a dangling pointer as the new stack head.
//
//   The name "ABA" describes the value sequence seen by the CAS target:
//     A (Thread 1 reads) → B (Thread 2 pops A) → A (Thread 2 pushes A back)
//   The CAS cannot distinguish "A was never changed" from "A changed and returned".
//
// head_ is public so the ABA scenario test can force a manual CAS.

template <typename T>
class BuggyStack {
public:
    struct Node {
        T     data;
        Node* next{nullptr};
    };

    alignas(64) std::atomic<Node*> head_{nullptr};

    void push(Node* n) noexcept {
        Node* old = head_.load(std::memory_order_acquire);
        do {
            n->next = old;
        } while (!head_.compare_exchange_weak(old, n,
                                              std::memory_order_release,
                                              std::memory_order_acquire));
    }

    // ABA window: between loading old and executing the CAS, another thread
    // can pop old, free it, and push a new node with the same address.
    // The CAS will succeed but install a stale or dangling next pointer.
    Node* pop() noexcept {
        Node* old = head_.load(std::memory_order_acquire);
        while (old) {
            Node* next = old->next;           // <-- ABA window begins
            if (head_.compare_exchange_weak(  // <-- ABA window ends
                    old, next,
                    std::memory_order_release,
                    std::memory_order_acquire))
                return old;
        }
        return nullptr;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == nullptr;
    }
};

} // namespace foundation
