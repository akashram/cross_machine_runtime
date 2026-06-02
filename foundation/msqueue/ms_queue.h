#pragma once

// MsQueue<T>: lock-free FIFO queue with hazard pointer memory reclamation.
//
// Based on: Michael & Scott, "Simple, Fast, and Practical Non-Blocking and
// Blocking Concurrent Queue Algorithms," PODC 1996.
//
// Structure:
//   head_ → [sentinel] → [first real] → ... → [last real] ← tail_
//
//   head_ always points to a sentinel (dummy) node whose data field is never
//   read. The first real value in the queue is head_->next.
//
//   tail_ points to the last node but may lag: when enqueue() commits a new
//   node by CAS-ing tail_->next from null to the new node, it has not yet
//   advanced tail_. Any thread that observes this lag (tail_->next != null)
//   helps advance tail_ before retrying its own operation.
//
// Enqueue (lock-free):
//   1. HP[0] = tail_  (protect from concurrent retire)
//   2. Load tail_->next.
//   3. If next != null: tail_ is lagging. CAS tail_ to next (help). Retry.
//   4. CAS tail_->next from null to new_node.  If fails, retry.
//   5. CAS tail_ to new_node (best-effort — another thread may win).
//
// Dequeue (lock-free):
//   1. HP[0] = head_  (protect sentinel)
//   2. HP[1] = head_->next  (protect first real element)
//      Both use the protect_raw + seq_cst re-read validation protocol.
//   3. Validate head_ == HP[0]. If changed, retry.
//   4. If next == null: queue is empty, return false.
//   5. If head_ == tail_: tail_ is lagging. Help advance. Retry.
//   6. Copy next->data into a local variable (before CAS).
//   7. CAS head_ from head to next. On success: retire(head), return true.
//
// Why copy data before CAS (step 6):
//   Multiple threads may reach step 6 for the same `next` node concurrently.
//   Moving from next->data would be a write concurrent with others' reads —
//   a C++ data race. Copying is safe (all concurrent accesses are reads).
//   Only the CAS winner's local copy is used; others discard their copies
//   when their CAS fails. Requires T to be copy-constructible.
//
// Why hazard pointers, not tagged pointers:
//   The MS queue is not susceptible to ABA via tagged pointers because:
//   - tail_->next is written from null exactly once (by the enqueue that
//     installs the node); it never returns to null. No ABA on the null CAS.
//   - head_ CAS: ABA requires thread A to see head = X, X to be dequeued and
//     FREED (address reused for a new node with same value), then head = X
//     again. HP[0] = X prevents freeing X, so its address cannot be reused.
//     ABA is structurally impossible with HP protection.
//
// Memory ordering:
//   enqueue CAS on tail_->next: release — ensures new_node->data writes
//     are visible to threads that load this next pointer with acquire.
//   dequeue CAS on head_: release/acquire — ensures the winner observes all
//     stores made by the enqueue that committed this node.
//   HP stores: seq_cst (per HazardDomain protocol — total order with scan()).
//
// Thread safety: all public methods are fully thread-safe.
// empty() is only accurate in quiescent state (see comment on the method).

#include "hazard/hazard_ptr.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace foundation {

template <typename T>
class MsQueue {
    static_assert(std::is_copy_constructible_v<T>,
                  "MsQueue<T> requires T to be copy-constructible "
                  "(data is copied before the CAS to avoid concurrent write races)");

    // data is optional<T> so the sentinel can be constructed without ever
    // touching T. The sentinel's data is nullopt and is never dereferenced.
    struct Node {
        std::optional<T>   data;
        std::atomic<Node*> next{nullptr};

        Node() = default;                                 // sentinel: data = nullopt
        explicit Node(T val) : data{std::move(val)} {}   // real node
    };

public:
    MsQueue() {
        Node* s = new Node{};                              // sentinel
        head_.store(s, std::memory_order_relaxed);
        tail_.store(s, std::memory_order_relaxed);
    }

    ~MsQueue() noexcept {
        while (drain_one()) {}
        delete head_.load(std::memory_order_relaxed);     // free final sentinel
        domain_.scan();                                   // drain pending retires
    }

    MsQueue(const MsQueue&)            = delete;
    MsQueue& operator=(const MsQueue&) = delete;

    // Enqueue a value. Allocates one node. Never blocks.
    void enqueue(T val) {
        auto* node = new Node{std::move(val)};
        HazardDomain::Guard hp(domain_, 0);

        while (true) {
            // Protect tail_ before dereferencing it.
            Node* tail;
            do {
                tail = tail_.load(std::memory_order_relaxed);
                hp.protect_raw(static_cast<void*>(tail));
            } while (tail_.load(std::memory_order_seq_cst) != tail);

            Node* next = tail->next.load(std::memory_order_acquire);

            if (tail != tail_.load(std::memory_order_acquire))
                continue;  // tail moved; re-protect

            if (next == nullptr) {
                // Tail is truly the last node. Try to append.
                if (tail->next.compare_exchange_weak(next, node,
                                                     std::memory_order_release,
                                                     std::memory_order_acquire)) {
                    // Advance tail_ (best-effort — may lose to another thread).
                    tail_.compare_exchange_strong(tail, node,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed);
                    return;
                }
            } else {
                // Tail is lagging. Help advance it before retrying.
                tail_.compare_exchange_strong(tail, next,
                                              std::memory_order_release,
                                              std::memory_order_relaxed);
            }
        }
    }

    // Dequeue the front value. Returns false if empty. Never blocks.
    bool dequeue(T& val) {
        HazardDomain::Guard hp0(domain_, 0);
        HazardDomain::Guard hp1(domain_, 1);

        while (true) {
            // Step 1: protect the sentinel node (head_).
            Node* head;
            do {
                head = head_.load(std::memory_order_relaxed);
                hp0.protect_raw(static_cast<void*>(head));
            } while (head_.load(std::memory_order_seq_cst) != head);

            // Step 2: protect head->next (the first real element).
            // head is protected (hp0), so accessing head->next is safe.
            Node* next;
            do {
                next = head->next.load(std::memory_order_relaxed);
                hp1.protect_raw(static_cast<void*>(next));
            } while (head->next.load(std::memory_order_seq_cst) != next);

            // Step 3: validate head_ is still our protected head.
            // Re-read with seq_cst to ensure total order with the HP stores above.
            if (head != head_.load(std::memory_order_seq_cst))
                continue;

            // Step 4: empty queue.
            if (next == nullptr) return false;

            Node* tail = tail_.load(std::memory_order_acquire);

            // Step 5: tail is lagging — help advance it.
            if (head == tail) {
                tail_.compare_exchange_strong(tail, next,
                                              std::memory_order_release,
                                              std::memory_order_relaxed);
                continue;
            }

            // Step 6: copy data before CAS (all concurrent accesses are reads).
            T tmp = *next->data;

            // Step 7: try to advance head_ from sentinel to next.
            if (head_.compare_exchange_weak(head, next,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
                val = std::move(tmp);
                hp0.reset();
                hp1.reset();
                domain_.retire(head);   // retire the old sentinel
                return true;
            }
        }
    }

    // Approximate empty check. Only accurate in quiescent state: a concurrent
    // enqueue may have committed a node to head_->next between the two loads.
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire)
                    ->next.load(std::memory_order_acquire) == nullptr;
    }

    // Force a hazard-pointer scan, freeing all pending retires.
    // Call after all threads have finished using the queue to verify
    // that all nodes have been reclaimed.
    void drain() noexcept { domain_.scan(); }

private:
    // Remove one node from the queue without materialising T.
    // Used by the destructor to drain remaining elements without requiring
    // T to be default-constructible.
    bool drain_one() noexcept {
        HazardDomain::Guard hp0(domain_, 0);
        HazardDomain::Guard hp1(domain_, 1);
        while (true) {
            Node* head;
            do {
                head = head_.load(std::memory_order_relaxed);
                hp0.protect_raw(static_cast<void*>(head));
            } while (head_.load(std::memory_order_seq_cst) != head);

            Node* next;
            do {
                next = head->next.load(std::memory_order_relaxed);
                hp1.protect_raw(static_cast<void*>(next));
            } while (head->next.load(std::memory_order_seq_cst) != next);

            if (head != head_.load(std::memory_order_seq_cst)) continue;
            if (next == nullptr) return false;

            Node* tail = tail_.load(std::memory_order_acquire);
            if (head == tail) {
                tail_.compare_exchange_strong(tail, next,
                                              std::memory_order_release,
                                              std::memory_order_relaxed);
                continue;
            }
            if (head_.compare_exchange_weak(head, next,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
                hp0.reset();
                hp1.reset();
                domain_.retire(head);
                return true;
            }
        }
    }

    HazardDomain               domain_;
    alignas(64) std::atomic<Node*> head_;   // sentinel (data ignored)
    alignas(64) std::atomic<Node*> tail_;   // last node (may lag)
};

} // namespace foundation
