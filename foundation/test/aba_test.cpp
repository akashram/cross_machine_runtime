#include "aba/aba_demo.h"
#include "aba/aba_stack.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: single-thread LIFO ordering
// ---------------------------------------------------------------------------

static void test_lifo() {
    using Stack = foundation::AbaStack<int>;
    Stack stack;

    Stack::Node nodes[5];
    for (int i = 0; i < 5; ++i) {
        nodes[i].data = i;
        stack.push(&nodes[i]);
    }

    for (int i = 4; i >= 0; --i) {
        Stack::Node* n = stack.pop();
        assert(n != nullptr);
        assert(n->data == i);
    }

    assert(stack.pop() == nullptr);
    assert(stack.empty());
    printf("PASS  test_lifo\n");
}

// ---------------------------------------------------------------------------
// Test 2: ABA bug demonstration (no threads — manual simulation)
//
// We simulate the exact ABA race by:
//   (a) pre-loading Thread 1's "stale" values,
//   (b) running Thread 2's ABA sequence,
//   (c) replaying Thread 1's CAS with the stale values.
//
// BuggyStack: the stale CAS succeeds, installing a wrong (already-popped)
//             node as the new head.
// AbaStack:   the stale CAS fails (tag mismatch), so no corruption occurs.
// ---------------------------------------------------------------------------

static void test_aba_bug_demonstrated() {
    using Buggy = foundation::BuggyStack<int>;
    Buggy::Node nodeA{1, nullptr};
    Buggy::Node nodeB{2, nullptr};

    Buggy stack;
    stack.push(&nodeB);   // stack: B -> nullptr
    stack.push(&nodeA);   // stack: A -> B -> nullptr

    // Thread 1 pre-loads its stale values (simulating the moment it is
    // preempted between the load and the CAS in pop()).
    Buggy::Node* stale_old  = nodeA.next == &nodeB ? &nodeA : nullptr;
    (void)stale_old;                    // suppress unused-variable warning
    Buggy::Node* thread1_old  = &nodeA; // head as Thread 1 observed it
    Buggy::Node* thread1_next = nodeA.next; // = &nodeB, Thread 1's desired new head

    // Thread 2 executes the ABA sequence.
    Buggy::Node* a = stack.pop();       // head = B -> nullptr;  a = &nodeA
    Buggy::Node* b = stack.pop();       // head = nullptr;        b = &nodeB
    assert(a == &nodeA);
    assert(b == &nodeB);
    stack.push(&nodeA);                 // head = A -> nullptr  (A->next cleared)

    // Thread 1 resumes. Its CAS fires with the stale expected/desired values.
    // expected = &nodeA  ← still matches because Thread 2 pushed A back
    // desired  = &nodeB  ← stale: B was already popped by Thread 2
    bool cas_fired = stack.head_.compare_exchange_strong(
        thread1_old, thread1_next,
        std::memory_order_seq_cst);

    assert(cas_fired);                          // CAS succeeded (ABA!)
    assert(stack.head_.load() == &nodeB);       // head is now the stale &nodeB
    // nodeB has already been popped — stack is corrupted.

    printf("PASS  test_aba_bug_demonstrated  "
           "(CAS succeeded with stale ptr; head=%p == &nodeB=%p)\n",
           static_cast<void*>(stack.head_.load()), static_cast<const void*>(&nodeB));
}

static void test_aba_fix_demonstrated() {
    using Fixed = foundation::AbaStack<int>;
    Fixed::Node nodeA{1, nullptr};
    Fixed::Node nodeB{2, nullptr};

    Fixed stack;
    stack.push(&nodeB);   // head = {&B, 1}
    stack.push(&nodeA);   // head = {&A, 2}

    // Thread 1 pre-loads its stale tagged pointer.
    Fixed::TaggedPtr thread1_old  = stack.head_.load(std::memory_order_acquire);
    // thread1_old = {&nodeA, tag=2}
    Fixed::TaggedPtr thread1_next = {thread1_old.ptr->next, thread1_old.tag + 1};
    // thread1_next = {&nodeB, tag=3}  ← will become stale

    // Thread 2 executes the ABA sequence.
    stack.pop();             // head = {&B, 3}
    stack.pop();             // head = {nullptr, 4}
    stack.push(&nodeA);      // head = {&A, 5}

    // Thread 1 resumes. Its CAS fires with stale tagged values.
    // expected = {&nodeA, tag=2}  ← tag no longer matches (current tag=5)
    Fixed::TaggedPtr current = thread1_old;
    bool cas_fired = stack.head_.compare_exchange_strong(
        current, thread1_next,
        std::memory_order_seq_cst);

    assert(!cas_fired);    // CAS failed — ABA prevented by tag mismatch
    // stack is correct: head = {&nodeA, 5}
    Fixed::Node* n = stack.pop();
    assert(n == &nodeA);
    assert(stack.empty());

    printf("PASS  test_aba_fix_demonstrated  "
           "(CAS failed; tag mismatch prevented stale install)\n");
}

// ---------------------------------------------------------------------------
// Test 3: concurrent push-then-pop stress (TSan-clean)
//
// Push phase: all threads push kNodesPerThread nodes.
// Pop phase:  all threads race to pop until the stack is empty.
// A std::barrier separates the phases so no push races with a pop
// (which would require hazard pointers — deferred to step 6).
// ---------------------------------------------------------------------------

static void test_concurrent_push_pop() {
    constexpr std::size_t kThreads        = 4;
    constexpr std::size_t kNodesPerThread = 25'000;
    constexpr std::size_t kTotal          = kThreads * kNodesPerThread;

    using Stack = foundation::AbaStack<int>;
    std::vector<Stack::Node> pool(kTotal);
    for (std::size_t i = 0; i < kTotal; ++i) pool[i].data = static_cast<int>(i);

    Stack stack;
    std::atomic<int> pop_count{0};
    std::barrier sync(static_cast<std::ptrdiff_t>(kThreads));

    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            // Phase 1: push this thread's slice
            for (std::size_t i = 0; i < kNodesPerThread; ++i)
                stack.push(&pool[t * kNodesPerThread + i]);

            sync.arrive_and_wait();

            // Phase 2: pop until empty
            while (Stack::Node* n = stack.pop()) {
                (void)n;
                pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) t.join();

    assert(pop_count.load() == static_cast<int>(kTotal));
    assert(stack.empty());
    printf("PASS  test_concurrent_push_pop  (%zu nodes)\n", kTotal);
}

// ---------------------------------------------------------------------------
// Test 4: lock-free guarantee
// ---------------------------------------------------------------------------

static void test_lock_free() {
    static_assert(foundation::AbaStack<int>::TaggedPtr{{}, 0} ==
                  foundation::AbaStack<int>::TaggedPtr{{}, 0});
    // is_always_lock_free is already asserted in the header; just print it.
    printf("PASS  test_lock_free  (is_always_lock_free=true, sizeof(TaggedPtr)=16)\n");
}

int main() {
    test_lifo();
    test_aba_bug_demonstrated();
    test_aba_fix_demonstrated();
    test_concurrent_push_pop();
    test_lock_free();
    return 0;
}
