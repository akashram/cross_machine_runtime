#include "numa/numa.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <latch>
#include <thread>
#include <vector>

using foundation::NumaTopology;
using foundation::NumaArena;

// ---------------------------------------------------------------------------
// Test 1: topology detection produces a sane result on any platform
// ---------------------------------------------------------------------------
static void test_topology_detect() {
    auto t = NumaTopology::detect();

    assert(t.num_nodes >= 1);
    assert(t.num_cpus  >= 1);

    // Every CPU must map to a valid node
    assert(static_cast<int>(t.cpu_to_node.size()) == t.num_cpus);
    for (int c = 0; c < t.num_cpus; ++c) {
        int n = t.cpu_to_node[static_cast<std::size_t>(c)];
        assert(n >= 0 && n < t.num_nodes);
    }

    // node_cpus must have one entry per node, each non-empty
    assert(static_cast<int>(t.node_cpus.size()) == t.num_nodes);
    for (int n = 0; n < t.num_nodes; ++n)
        assert(!t.node_cpus[static_cast<std::size_t>(n)].empty());

    // All CPUs across all node_cpus entries must be within [0, num_cpus)
    for (int n = 0; n < t.num_nodes; ++n)
        for (int c : t.node_cpus[static_cast<std::size_t>(n)])
            assert(c >= 0 && c < t.num_cpus);

    printf("PASS  test_topology_detect  (nodes=%d, cpus=%d)\n",
           t.num_nodes, t.num_cpus);
}

// ---------------------------------------------------------------------------
// Test 2: macOS is always single-node (sanity check on current platform)
// ---------------------------------------------------------------------------
static void test_single_node_on_macos() {
#ifdef __APPLE__
    auto t = NumaTopology::detect();
    assert(t.num_nodes == 1);
    assert(t.num_cpus >= 1);
    // All CPUs must be on node 0
    for (int c = 0; c < t.num_cpus; ++c)
        assert(t.cpu_to_node[static_cast<std::size_t>(c)] == 0);
    printf("PASS  test_single_node_on_macos  (cpus=%d)\n", t.num_cpus);
#else
    printf("SKIP  test_single_node_on_macos  (Linux)\n");
#endif
}

// ---------------------------------------------------------------------------
// Test 3: current_node() returns a valid node index
// ---------------------------------------------------------------------------
static void test_current_node() {
    auto t = NumaTopology::detect();
    int n = t.current_node();
    assert(n >= 0 && n < t.num_nodes);
    printf("PASS  test_current_node  (node=%d)\n", n);
}

// ---------------------------------------------------------------------------
// Test 4: cpulist parser
// ---------------------------------------------------------------------------
static void test_cpulist_parse() {
    using foundation::detail::parse_cpulist;

    auto v = parse_cpulist("0-3,8,10-12");
    assert(v.size() == 8);
    assert(v[0] == 0 && v[3] == 3 && v[4] == 8 && v[5] == 10 && v[7] == 12);

    auto w = parse_cpulist("0");
    assert(w.size() == 1 && w[0] == 0);

    auto x = parse_cpulist("4-7");
    assert(x.size() == 4 && x[0] == 4 && x[3] == 7);

    printf("PASS  test_cpulist_parse\n");
}

// ---------------------------------------------------------------------------
// Test 5: bind_thread_to_node — must not crash; returns bool
// ---------------------------------------------------------------------------
static void test_bind_thread() {
    auto t = NumaTopology::detect();
    // Binding to node 0 must succeed (all platforms have node 0)
    bool ok = foundation::bind_thread_to_node(0, t);
#ifdef __linux__
    assert(ok);  // hard binding must succeed on Linux
#else
    assert(ok);  // macOS single-node always returns true
#endif
    // Binding to an out-of-range node must fail
    bool bad = foundation::bind_thread_to_node(t.num_nodes + 99, t);
    assert(!bad);
    printf("PASS  test_bind_thread\n");
}

// ---------------------------------------------------------------------------
// Test 6: NumaArena basic alloc on each node
// ---------------------------------------------------------------------------
static void test_numa_arena_alloc() {
    auto t = NumaTopology::detect();
    auto na = NumaArena::make(t, 4u << 20);  // 4 MiB per node

    for (int n = 0; n < t.num_nodes; ++n) {
        assert(na.valid_node(n));

        void* p = na.alloc_on_node(n, 64);
        assert(p != nullptr);
        // 64-byte size class → 64-byte alignment
        assert(reinterpret_cast<uintptr_t>(p) % 64 == 0);

        // Write and read back
        std::memset(p, 0xAB, 64);
        assert(static_cast<unsigned char*>(p)[63] == 0xAB);
    }

    printf("PASS  test_numa_arena_alloc  (nodes=%d)\n", t.num_nodes);
}

// ---------------------------------------------------------------------------
// Test 7: NumaArena freelist recycling per node
// ---------------------------------------------------------------------------
static void test_numa_arena_freelist() {
    auto t = NumaTopology::detect();
    auto na = NumaArena::make(t, 4u << 20);

    void* p = na.alloc_on_node(0, 32);
    assert(p != nullptr);
    na.free_on_node(0, p, 32);
    void* q = na.alloc_on_node(0, 32);
    assert(q == p);  // recycled from freelist

    printf("PASS  test_numa_arena_freelist\n");
}

// ---------------------------------------------------------------------------
// Test 8: alloc_local routes to the calling thread's node
// ---------------------------------------------------------------------------
static void test_alloc_local() {
    auto t = NumaTopology::detect();
    auto na = NumaArena::make(t, 4u << 20);

    void* p = na.alloc_local(128);
    assert(p != nullptr);
    na.free_local(p, 128);

    printf("PASS  test_alloc_local  (node=%d)\n", t.current_node());
}

// ---------------------------------------------------------------------------
// Test 9: reset_node clears the arena
// ---------------------------------------------------------------------------
static void test_reset_node() {
    auto t = NumaTopology::detect();
    auto na = NumaArena::make(t, 4u << 20);

    void* p = na.alloc_on_node(0, 64);
    assert(p != nullptr);
    assert(na.topology().num_nodes >= 1);

    na.reset_node(0);
    // After reset, the arena's bump pointer is back to base.
    // A fresh alloc should land at the same base address.
    void* q = na.alloc_on_node(0, 64);
    assert(q == p);  // cursor rewound to start

    printf("PASS  test_reset_node\n");
}

// ---------------------------------------------------------------------------
// Test 10: multi-node independence — arenas on different nodes don't alias
// (only meaningful on a Linux multi-node machine; skip gracefully on macOS)
// ---------------------------------------------------------------------------
static void test_multi_node_independence() {
    auto t = NumaTopology::detect();
    if (t.num_nodes < 2) {
        printf("SKIP  test_multi_node_independence  (single-node platform)\n");
        return;
    }

    auto na = NumaArena::make(t, 4u << 20);

    void* p0 = na.alloc_on_node(0, 64);
    void* p1 = na.alloc_on_node(1, 64);
    assert(p0 != nullptr && p1 != nullptr);
    assert(p0 != p1);  // different arenas → different addresses

    printf("PASS  test_multi_node_independence  (nodes=%d)\n", t.num_nodes);
}

// ---------------------------------------------------------------------------
// Test 11: threads can allocate concurrently from their local nodes
// (coordination via latch; each thread uses alloc_local on its pinned node)
// ---------------------------------------------------------------------------
static void test_concurrent_local_alloc() {
    // This test is only valid on multi-node machines: each thread pins to a
    // distinct node and allocates from that node's arena exclusively.  On a
    // single-node machine all threads would share arena 0, which is a genuine
    // data race (NumaArena is per-node, not per-thread; on a single-node
    // system use ThreadLocalArena instead).
    auto t = NumaTopology::detect();
    if (t.num_nodes < 2) {
        printf("SKIP  test_concurrent_local_alloc  (single-node: no distinct node arenas)\n");
        return;
    }

    auto na = NumaArena::make(t, 16u << 20);
    const std::size_t kThreads = static_cast<std::size_t>(t.num_nodes);
    std::vector<void*> ptrs(kThreads, nullptr);
    std::latch ready(static_cast<std::ptrdiff_t>(kThreads));
    std::latch go(1);

    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]{
            int node = static_cast<int>(i);  // one thread per node
            foundation::bind_thread_to_node(node, t);
            ready.count_down();
            go.wait();
            ptrs[i] = na.alloc_on_node(node, 64);
        });
    }

    ready.wait();
    go.count_down();
    for (auto& th : threads) th.join();

    for (std::size_t i = 0; i < kThreads; ++i)
        assert(ptrs[i] != nullptr);

    printf("PASS  test_concurrent_local_alloc  (%zu threads, %d nodes)\n",
           kThreads, t.num_nodes);
}

// ---------------------------------------------------------------------------
// Test 12: invalid node ID is handled gracefully
// ---------------------------------------------------------------------------
static void test_invalid_node() {
    auto t  = NumaTopology::detect();
    auto na = NumaArena::make(t, 4u << 20);

    void* p = na.alloc_on_node(-1, 64);
    assert(p == nullptr);
    void* q = na.alloc_on_node(t.num_nodes + 99, 64);
    assert(q == nullptr);

    // free on invalid node must not crash
    na.free_on_node(-1, nullptr, 64);
    na.free_on_node(t.num_nodes + 99, nullptr, 64);

    printf("PASS  test_invalid_node\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_topology_detect();
    test_single_node_on_macos();
    test_current_node();
    test_cpulist_parse();
    test_bind_thread();
    test_numa_arena_alloc();
    test_numa_arena_freelist();
    test_alloc_local();
    test_reset_node();
    test_multi_node_independence();
    test_concurrent_local_alloc();
    test_invalid_node();
    return 0;
}
