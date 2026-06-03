#include "prefetch/prefetch.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using cpu_engine::PrefetchHint;
using cpu_engine::prefetch_r;
using cpu_engine::prefetch_w;
using cpu_engine::prefetch_ahead;
using cpu_engine::prefetch_ahead_w;
using cpu_engine::make_pointer_chase_list;

// ---------------------------------------------------------------------------
// Test 1: prefetch_r with all hints does not crash
// ---------------------------------------------------------------------------
static void test_prefetch_r_hints() {
    static const int data[64] = {};

    prefetch_r<PrefetchHint::T0> (data);
    prefetch_r<PrefetchHint::T1> (data);
    prefetch_r<PrefetchHint::T2> (data);
    prefetch_r<PrefetchHint::NTA>(data);

    printf("PASS  test_prefetch_r_hints  (T0/T1/T2/NTA)\n");
}

// ---------------------------------------------------------------------------
// Test 2: prefetch_w with all hints does not crash
// ---------------------------------------------------------------------------
static void test_prefetch_w_hints() {
    static int data[64] = {};

    prefetch_w<PrefetchHint::T0> (data);
    prefetch_w<PrefetchHint::T1> (data);
    prefetch_w<PrefetchHint::T2> (data);
    prefetch_w<PrefetchHint::NTA>(data);

    printf("PASS  test_prefetch_w_hints  (T0/T1/T2/NTA)\n");
}

// ---------------------------------------------------------------------------
// Test 3: prefetch_ahead does not change computational result
//
// Prefetch is a pure hint — the computed sum must be identical with and
// without prefetching. If it differs, the prefetch wrappers have a bug.
// ---------------------------------------------------------------------------
static void test_prefetch_ahead_correctness() {
    constexpr std::size_t N    = 4096;
    constexpr std::size_t DIST = 16;

    std::vector<int> arr(N);
    std::iota(arr.begin(), arr.end(), 0);

    // Sum without prefetch
    int64_t sum_no_pf = 0;
    for (std::size_t i = 0; i < N; ++i)
        sum_no_pf += arr[i];

    // Sum with T0 prefetch
    int64_t sum_t0 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (i + DIST < N) prefetch_ahead(arr.data(), i, DIST);
        sum_t0 += arr[i];
    }

    // Sum with NTA prefetch
    int64_t sum_nta = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (i + DIST < N) prefetch_ahead<int, PrefetchHint::NTA>(arr.data(), i, DIST);
        sum_nta += arr[i];
    }

    assert(sum_no_pf == sum_t0);
    assert(sum_no_pf == sum_nta);

    int64_t expected = static_cast<int64_t>(N) * (N - 1) / 2;
    assert(sum_no_pf == expected);

    printf("PASS  test_prefetch_ahead_correctness  (sum=%lld  N=%zu  dist=%zu)\n",
           static_cast<long long>(sum_no_pf), N, DIST);
}

// ---------------------------------------------------------------------------
// Test 4: prefetch_ahead_w does not change result
// ---------------------------------------------------------------------------
static void test_prefetch_ahead_w_correctness() {
    constexpr std::size_t N    = 1024;
    constexpr std::size_t DIST = 8;

    std::vector<int> arr(N, 1);

    int64_t sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (i + DIST < N) prefetch_ahead_w(arr.data(), i, DIST);
        arr[i] *= 2;
        sum += arr[i];
    }

    assert(sum == static_cast<int64_t>(N) * 2);
    printf("PASS  test_prefetch_ahead_w_correctness  (N=%zu)\n", N);
}

// ---------------------------------------------------------------------------
// Test 5: make_pointer_chase_list visits every element exactly once
// ---------------------------------------------------------------------------
static void test_pointer_chase_list_coverage() {
    constexpr std::size_t N = 1024;
    std::vector<std::size_t> buf(N);
    make_pointer_chase_list(buf.data(), N);

    // Walk from 0, count visits
    std::vector<int> visited(N, 0);
    std::size_t idx = 0;
    for (std::size_t step = 0; step < N; ++step) {
        assert(idx < N);
        assert(visited[idx] == 0);   // must not revisit
        visited[idx] = 1;
        idx = buf[idx];
    }
    assert(idx == 0);  // cyclic: must return to start after N steps

    // Every element visited exactly once
    for (std::size_t i = 0; i < N; ++i)
        assert(visited[i] == 1);

    printf("PASS  test_pointer_chase_list_coverage  (N=%zu, all elements visited)\n", N);
}

// ---------------------------------------------------------------------------
// Test 6: make_pointer_chase_list is not the identity permutation (random)
// ---------------------------------------------------------------------------
static void test_pointer_chase_list_is_random() {
    constexpr std::size_t N = 256;
    std::vector<std::size_t> buf(N);
    make_pointer_chase_list(buf.data(), N);

    // Count how many buf[i] == (i+1) % N (sequential walk — should be rare)
    int sequential_steps = 0;
    for (std::size_t i = 0; i < N; ++i)
        if (buf[i] == (i + 1) % N) ++sequential_steps;

    // With a random permutation of 256 elements, expect ~1 sequential step on average.
    // Allow up to 10 to be robust to random seeds.
    assert(sequential_steps < 10);

    printf("PASS  test_pointer_chase_list_is_random  "
           "(sequential steps: %d / %zu)\n", sequential_steps, N);
}

// ---------------------------------------------------------------------------
// Test 7: different seeds produce different permutations
// ---------------------------------------------------------------------------
static void test_pointer_chase_list_seed_variation() {
    constexpr std::size_t N = 128;
    std::vector<std::size_t> a(N), b(N);
    make_pointer_chase_list(a.data(), N, 0x111111111111111ull);
    make_pointer_chase_list(b.data(), N, 0x999999999999999ull);

    int differences = 0;
    for (std::size_t i = 0; i < N; ++i)
        if (a[i] != b[i]) ++differences;

    assert(static_cast<std::size_t>(differences) > N / 2);  // at least half should differ
    printf("PASS  test_pointer_chase_list_seed_variation  "
           "(%d / %zu positions differ)\n", differences, N);
}

// ---------------------------------------------------------------------------
// Test 8: prefetch_r on nullptr does not crash (implementation-defined but
// practically safe — the prefetch instruction is speculative and the CPU
// silently ignores faults during speculative loads)
// ---------------------------------------------------------------------------
static void test_prefetch_nullptr() {
    // On x86, PREFETCH* with an invalid address is a no-op (the speculative
    // load is silently aborted). On some ARM implementations it may also be a
    // no-op. This test documents and verifies this behaviour.
    prefetch_r(nullptr);
    printf("PASS  test_prefetch_nullptr  (no crash on invalid address)\n");
}

int main() {
    test_prefetch_r_hints();
    test_prefetch_w_hints();
    test_prefetch_ahead_correctness();
    test_prefetch_ahead_w_correctness();
    test_pointer_chase_list_coverage();
    test_pointer_chase_list_is_random();
    test_pointer_chase_list_seed_variation();
    test_prefetch_nullptr();
    printf("\nAll prefetch tests passed.\n");
}
