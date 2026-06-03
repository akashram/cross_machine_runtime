#include "proptest/proptest.h"

// Data structures under test
#include "spsc_queue.h"
#include "mpmc_queue.h"
#include "freelist/freelist.h"
#include "msqueue/ms_queue.h"
#include "arena/arena.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <latch>
#include <numeric>
#include <thread>
#include <vector>

using proptest::check;
using proptest::check_int;
using proptest::check_intvec;
using proptest::gen_int;
using proptest::gen_int_biased;
using proptest::gen_nonempty_vector;
using proptest::gen_size;
using proptest::gen_vector;
using proptest::shrink;
using proptest::CheckResult;

static int g_fails = 0;
static void require(const CheckResult& r, const char* name) {
    if (!r.passed) { ++g_fails; (void)name; }
}

// ===========================================================================
// Section 0: Framework self-tests (verify the framework works before trusting it)
// ===========================================================================

static void test_framework_rng() {
    // Splitmix64 is deterministic: same seed → same sequence.
    proptest::Rng a(12345), b(12345);
    for (int i = 0; i < 100; ++i)
        assert(a.next() == b.next());

    // Different seeds produce different sequences.
    proptest::Rng c(99999);
    int diff_count = 0;
    for (int i = 0; i < 20; ++i)
        if (a.next() != c.next()) ++diff_count;
    if (diff_count == 0) __builtin_trap();

    // next_int stays within bounds.
    proptest::Rng rng(1);
    for (int i = 0; i < 1000; ++i) {
        int v = static_cast<int>(rng.next_int(-50, 50));
        if (v < -50 || v > 50) __builtin_trap();
    }
    printf("PASS  framework: rng\n");
}

static void test_framework_shrink_int() {
    // shrink(0) must be empty (can't go smaller).
    assert(shrink(0).empty());

    // All shrinks of n are strictly "closer to zero".
    for (int n : {-100, -1, 1, 100, 42}) {
        for (int s : shrink(n))
            if (!(std::abs(s) < std::abs(n) || (n < 0 && s > n) || (n > 0 && s < n)))
                __builtin_trap();
    }

    // shrink finds a minimal counterexample.
    // Property: all ints are non-negative. Counterexample: -1.
    auto result = check_int("shrink_demo",
        gen_int(-10, 10),
        [](int v) { return v >= 0; });
    // Should fail and shrink to -1
    assert(!result.passed);
    assert(result.failure_info.find("-1") != std::string::npos);
    printf("PASS  framework: int shrinking (shrunk to -1 as expected)\n");
}

static void test_framework_vector_shrink() {
    // A vector property that fails for any vector containing -1.
    auto result = check_intvec("vector_contains_neg1",
        gen_vector(gen_int(-5, 5), 10),
        [](const std::vector<int>& v) {
            return std::find(v.begin(), v.end(), -1) == v.end();
        });
    assert(!result.passed);
    // Shrunk result should be a 1-element vector [-1].
    assert(result.failure_info.find("-1") != std::string::npos);
    printf("PASS  framework: vector shrinking\n");
}

// ===========================================================================
// Section 1: SPSC queue properties
// ===========================================================================

// SpscQueue<T,N> has compile-time capacity N-1 (one slot is a sentinel).
// Use a fixed large capacity; property inputs must fit within it.
static constexpr std::size_t kSpscCap = 256;

// Property: push N items then pop N items → FIFO order preserved.
// Invariant: for any sequence V, dequeue(enqueue_all(V)) == V.
static void prop_spsc_fifo() {
    auto result = check("spsc: push-then-pop preserves FIFO",
        gen_nonempty_vector(gen_int_biased(), kSpscCap - 1),
        [](const std::vector<int>& pushed) -> bool {
            foundation::SpscQueue<int, kSpscCap> q;
            for (int v : pushed)
                if (!q.push(v)) return true;  // queue full — skip this input

            std::vector<int> popped;
            popped.reserve(pushed.size());
            int val;
            while (q.pop(val)) popped.push_back(val);

            return popped == pushed;
        });
    require(result, "spsc_fifo");
}

// Property: no items are lost across a concurrent producer/consumer pair.
// Invariant: count(pushed) == count(popped).
static void prop_spsc_no_loss() {
    // SpscQueue capacity is compile-time; use a fixed size.
    constexpr int kN = 500;
    foundation::SpscQueue<int, 512> q;
    std::atomic<int> consumed{0};

    std::thread producer([&]{
        for (int i = 0; i < kN; ++i)
            while (!q.push(i)) std::this_thread::yield();
    });
    std::thread consumer([&]{
        int val, got = 0;
        while (got < kN) {
            if (q.pop(val)) ++got;
            else std::this_thread::yield();
        }
        consumed.store(got, std::memory_order_relaxed);
    });
    producer.join();
    consumer.join();

    bool ok = consumed.load() == kN;
    if (ok) printf("OK    %-44s  %d items\n", "spsc: no items lost", kN);
    else { printf("FAIL  spsc: no items lost\n"); ++g_fails; }
}

// ===========================================================================
// Section 2: MPMC queue properties
// ===========================================================================

// Property: K producers × M items each, K consumers × M items each.
// Invariant: every pushed item is popped exactly once.
static void prop_mpmc_no_loss() {
    // Fixed parameters (concurrent tests can't easily vary size via gen)
    constexpr int kProducers = 3;
    constexpr int kConsumers = 3;
    constexpr int kPerThread = 200;
    constexpr int kTotal     = kProducers * kPerThread;

    foundation::MpmcQueue<int, 512> q;
    std::atomic<int> total_consumed{0};
    std::latch done(kConsumers);

    std::vector<std::thread> producers, consumers;
    for (int t = 0; t < kProducers; ++t) {
        producers.emplace_back([&, t]{
            for (int i = 0; i < kPerThread; ++i)
                while (!q.push(t * kPerThread + i)) std::this_thread::yield();
        });
    }
    for (int t = 0; t < kConsumers; ++t) {
        consumers.emplace_back([&]{
            int val;
            // Exit when total_consumed reaches kTotal (visible to all consumers).
            while (total_consumed.load(std::memory_order_acquire) < kTotal) {
                if (q.pop(val))
                    total_consumed.fetch_add(1, std::memory_order_release);
                else
                    std::this_thread::yield();
            }
            done.count_down();
        });
    }
    for (auto& p : producers) p.join();
    done.wait();
    for (auto& c : consumers) c.join();

    bool ok = (total_consumed.load() == kTotal);
    if (ok) printf("OK    %-44s  %d items\n", "mpmc: no items lost", kTotal);
    else { printf("FAIL  mpmc: no items lost\n"); ++g_fails; }
}

// ===========================================================================
// Section 3: Lock-free freelist properties
// ===========================================================================

struct FreelistNode { int value; };

// Property: acquire/release cycles never exceed the pool size.
// The pool has N slots; we do R acquire/release cycles and check that
// in-flight count never exceeds N.
static void prop_freelist_bounded() {
    auto result = check("freelist: in-flight count never exceeds pool size",
        gen_size(1, 32),
        [](std::size_t pool_size) -> bool {
            foundation::FreeList<FreelistNode> fl(pool_size);
            // Acquire all, verify we can't acquire more
            std::vector<FreelistNode*> held;
            held.reserve(pool_size);
            for (std::size_t i = 0; i < pool_size; ++i) {
                auto* p = fl.acquire();
                if (!p) return false;  // should get pool_size nodes
                held.push_back(p);
            }
            // Pool exhausted: next acquire must fail (return nullptr)
            if (fl.acquire() != nullptr) return false;
            // Release all
            for (auto* p : held) fl.release(p);
            // Should be able to acquire again
            auto* p = fl.acquire();
            if (!p) return false;
            fl.release(p);
            return true;
        });
    require(result, "freelist_bounded");
}

// Property: every released node is re-acquirable (no elements disappear).
static void prop_freelist_no_loss() {
    auto result = check("freelist: released nodes are reacquirable",
        gen_size(1, 16),
        [](std::size_t pool_size) -> bool {
            foundation::FreeList<FreelistNode> fl(pool_size);
            for (int round = 0; round < 5; ++round) {
                std::vector<FreelistNode*> held;
                for (std::size_t i = 0; i < pool_size; ++i) {
                    auto* p = fl.acquire();
                    if (!p) return false;
                    held.push_back(p);
                }
                for (auto* p : held) fl.release(p);
            }
            return true;
        });
    require(result, "freelist_no_loss");
}

// ===========================================================================
// Section 4: Michael-Scott queue properties
// ===========================================================================

// Property: single producer / single consumer preserves FIFO order.
static void prop_msqueue_fifo() {
    auto result = check("msqueue: push-then-pop preserves FIFO",
        gen_nonempty_vector(gen_int(-500, 500), 50),
        [](const std::vector<int>& pushed) -> bool {
            foundation::MsQueue<int> q;
            for (int v : pushed) q.enqueue(v);

            std::vector<int> popped;
            popped.reserve(pushed.size());
            int val;
            while (q.dequeue(val)) popped.push_back(val);

            return popped == pushed;
        });
    require(result, "msqueue_fifo");
}

// Property: no items lost in concurrent producer/consumer scenario.
static void prop_msqueue_no_loss() {
    constexpr int kItems = 500;
    foundation::MsQueue<int> q;
    std::atomic<int> popped_count{0};

    std::thread producer([&]{
        for (int i = 0; i < kItems; ++i) q.enqueue(i);
    });
    std::thread consumer([&]{
        int val, got = 0;
        while (got < kItems) {
            if (q.dequeue(val)) ++got;
            else std::this_thread::yield();
        }
        popped_count.store(got, std::memory_order_relaxed);
    });
    producer.join();
    consumer.join();

    bool ok = (popped_count.load() == kItems);
    if (ok) printf("OK    %-44s  %d items\n", "msqueue: no items lost", kItems);
    else { printf("FAIL  msqueue: no items lost\n"); ++g_fails; }
}

// ===========================================================================
// Section 5: Arena allocator properties
// ===========================================================================

// Property: all returned pointers are non-overlapping and within the slab.
static void prop_arena_nonoverlap() {
    auto result = check("arena: allocs are non-overlapping and in-slab",
        gen_nonempty_vector(gen_size(1, 256), 32),
        [](const std::vector<std::size_t>& sizes) -> bool {
            foundation::Arena a(1u << 20);  // 1 MiB
            std::vector<std::pair<char*, std::size_t>> allocs;

            for (std::size_t sz : sizes) {
                void* p = a.alloc(sz, 1);
                if (!p) break;  // ran out of space — skip remaining
                allocs.emplace_back(static_cast<char*>(p), sz);

                // Must be within the slab
                if (!a.owns(p)) return false;
            }

            // Check non-overlap: [p, p+sz) must not intersect any other range
            for (std::size_t i = 0; i < allocs.size(); ++i) {
                for (std::size_t j = i + 1; j < allocs.size(); ++j) {
                    char* ai = allocs[i].first, *bi = ai + allocs[i].second;
                    char* aj = allocs[j].first, *bj = aj + allocs[j].second;
                    if (ai < bj && aj < bi) return false;  // overlap
                }
            }
            return true;
        });
    require(result, "arena_nonoverlap");
}

// Property: reset rewinds to base; subsequent allocs reuse the same addresses.
static void prop_arena_reset_rewinds() {
    auto result = check("arena: reset rewinds cursor to base",
        gen_size(1, 64),
        [](std::size_t n) -> bool {
            foundation::Arena a(64 * 1024);
            std::vector<void*> first_pass;
            for (std::size_t i = 0; i < n; ++i) {
                void* p = a.alloc(16, 16);
                if (!p) return true;  // too many allocs — skip
                first_pass.push_back(p);
            }
            a.reset();
            std::vector<void*> second_pass;
            for (std::size_t i = 0; i < n; ++i) {
                void* p = a.alloc(16, 16);
                if (!p) return false;  // must succeed again after reset
                second_pass.push_back(p);
            }
            return first_pass == second_pass;  // same addresses reused
        });
    require(result, "arena_reset_rewinds");
}

// ===========================================================================
// Section 6: TensorHandle properties
// ===========================================================================

// Generator for valid shapes (rank 0–4, each dim 1–8).
static proptest::Gen<std::vector<int64_t>> gen_shape() {
    return proptest::Gen<std::vector<int64_t>>{
        [](proptest::Rng& rng, std::size_t c) -> std::vector<int64_t> {
            std::size_t rank = rng.next_size(0, std::min(std::size_t{4}, c / 5 + 1));
            std::vector<int64_t> shape;
            shape.reserve(rank);
            for (std::size_t i = 0; i < rank; ++i)
                shape.push_back(static_cast<int64_t>(rng.next_size(1, 8)));
            return shape;
        }};
}

// Property: numel == product(shape).
static void prop_tensor_numel() {
    auto result = check("tensor: numel == product(shape)",
        gen_shape(),
        [](const std::vector<int64_t>& shape) -> bool {
            auto t = foundation::TensorHandle::empty(shape, foundation::Dtype::kFloat32);
            if (!t.valid()) return true;
            int64_t expected = 1;
            for (auto s : shape) expected *= s;
            return t.numel() == expected;
        });
    require(result, "tensor_numel");
}

// Property: C-contiguous strides satisfy stride[i] == stride[i+1] * shape[i+1].
static void prop_tensor_contiguous_strides() {
    auto result = check("tensor: C-contiguous strides are correct",
        gen_shape(),
        [](const std::vector<int64_t>& shape) -> bool {
            auto t = foundation::TensorHandle::empty(shape, foundation::Dtype::kFloat32);
            if (!t.valid() || t.ndim() < 2) return true;
            for (int i = 0; i < t.ndim() - 1; ++i) {
                int64_t expected = t.stride(i + 1) * t.size(i + 1);
                if (t.stride(i) != expected) return false;
            }
            // Innermost stride == dtype_size
            if (t.stride(t.ndim() - 1) != 4) return false;  // float32 = 4B
            return true;
        });
    require(result, "tensor_contiguous_strides");
}

// Property: transpose(d0, d1).transpose(d0, d1) == original shape and strides.
// (transpose is its own inverse)
static void prop_tensor_transpose_involution() {
    // Use 2-d shapes for simplicity
    auto gen_2d = proptest::Gen<std::vector<int64_t>>{
        [](proptest::Rng& rng, std::size_t c) -> std::vector<int64_t> {
            int64_t m = static_cast<int64_t>(rng.next_size(1, std::max(std::size_t{1}, c/4 + 1)));
            int64_t n = static_cast<int64_t>(rng.next_size(1, std::max(std::size_t{1}, c/4 + 1)));
            return {m, n};
        }};

    auto result = check("tensor: transpose(0,1).transpose(0,1) == identity",
        gen_2d,
        [](const std::vector<int64_t>& shape) -> bool {
            auto t  = foundation::TensorHandle::empty(shape, foundation::Dtype::kFloat32);
            auto t2 = t.transpose(0, 1).transpose(0, 1);
            return t.shape() == t2.shape() && t.strides() == t2.strides();
        });
    require(result, "tensor_transpose_involution");
}

// Property: reshape to different valid shape preserves all element values.
static void prop_tensor_reshape_preserves_data() {
    // Generate a shape and a compatible reshaped shape with same numel
    auto gen_pair = proptest::Gen<std::pair<std::vector<int64_t>, std::vector<int64_t>>>{
        [](proptest::Rng& rng, std::size_t c) {
            std::size_t rank = rng.next_size(1, 3);
            int64_t numel = 1;
            std::vector<int64_t> shape;
            for (std::size_t i = 0; i < rank; ++i) {
                int64_t d = static_cast<int64_t>(rng.next_size(1, std::max(std::size_t{1}, c/4+1)));
                numel *= d;
                shape.push_back(d);
            }
            // New shape: flatten to [numel] (always valid)
            std::vector<int64_t> new_shape = {numel};
            return std::make_pair(shape, new_shape);
            (void)c;
        }};

    auto result = check("tensor: reshape preserves element data",
        gen_pair,
        [](const std::pair<std::vector<int64_t>, std::vector<int64_t>>& p) -> bool {
            const auto& [shape, new_shape] = p;
            auto t = foundation::TensorHandle::empty(shape, foundation::Dtype::kInt32);
            if (!t.valid()) return true;

            // Fill with sequential values
            int32_t* raw = t.data_as<int32_t>();
            for (int64_t i = 0; i < t.numel(); ++i) raw[i] = static_cast<int32_t>(i);

            auto r = t.reshape(new_shape);
            if (r.numel() != t.numel()) return false;

            // Verify data is unchanged (same pointer, same values)
            if (r.data() != t.data()) return false;
            for (int64_t i = 0; i < r.numel(); ++i)
                if (r.data_as<int32_t>()[i] != static_cast<int32_t>(i)) return false;
            return true;
        });
    require(result, "tensor_reshape_data");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== Framework self-tests ===\n");
    test_framework_rng();
    test_framework_shrink_int();
    test_framework_vector_shrink();

    printf("\n=== SPSC queue ===\n");
    prop_spsc_fifo();
    prop_spsc_no_loss();

    printf("\n=== MPMC queue ===\n");
    prop_mpmc_no_loss();

    printf("\n=== Lock-free freelist ===\n");
    prop_freelist_bounded();
    prop_freelist_no_loss();

    printf("\n=== Michael-Scott queue ===\n");
    prop_msqueue_fifo();
    prop_msqueue_no_loss();

    printf("\n=== Arena allocator ===\n");
    prop_arena_nonoverlap();
    prop_arena_reset_rewinds();

    printf("\n=== TensorHandle ===\n");
    prop_tensor_numel();
    prop_tensor_contiguous_strides();
    prop_tensor_transpose_involution();
    prop_tensor_reshape_preserves_data();

    printf("\n");
    if (g_fails == 0)
        printf("All property tests passed.\n");
    else
        printf("%d property test(s) FAILED.\n", g_fails);

    return g_fails == 0 ? 0 : 1;
}
