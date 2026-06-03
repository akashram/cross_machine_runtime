#include "branchless/branchless.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cmath>

using namespace cpu_engine;

// ---------------------------------------------------------------------------
// Test 1: branchless_select — basic ternary semantics
// ---------------------------------------------------------------------------
static void test_select() {
    assert(branchless_select(true,  1, 2) == 1);
    assert(branchless_select(false, 1, 2) == 2);
    assert(branchless_select(true,  1.5f, 2.5f) == 1.5f);
    assert(branchless_select(false, 1.5f, 2.5f) == 2.5f);

    // Pointer select
    int a = 10, b = 20;
    int* pa = &a; int* pb = &b;
    assert(branchless_select(true,  pa, pb) == &a);
    assert(branchless_select(false, pa, pb) == &b);

    printf("PASS  test_select\n");
}

// ---------------------------------------------------------------------------
// Test 2: branchless_select_bits — bit-manipulation path (int32 and int64)
// ---------------------------------------------------------------------------
static void test_select_bits() {
    assert(branchless_select_bits(true,  int32_t{10},  int32_t{20})  == 10);
    assert(branchless_select_bits(false, int32_t{10},  int32_t{20})  == 20);
    assert(branchless_select_bits(true,  int64_t{-5},  int64_t{99})  == -5);
    assert(branchless_select_bits(false, int64_t{-5},  int64_t{99})  == 99);
    assert(branchless_select_bits(true,  uint32_t{0},  uint32_t{42}) == 0u);
    assert(branchless_select_bits(false, uint32_t{0},  uint32_t{42}) == 42u);
    printf("PASS  test_select_bits\n");
}

// ---------------------------------------------------------------------------
// Test 3: branchless_min / branchless_max
// ---------------------------------------------------------------------------
static void test_min_max() {
    // int32
    assert(branchless_min(3,  7)  == 3);
    assert(branchless_min(-3, 7)  == -3);
    assert(branchless_min(7,  7)  == 7);
    assert(branchless_max(3,  7)  == 7);
    assert(branchless_max(-3, 7)  == 7);
    assert(branchless_max(7,  7)  == 7);

    // float
    assert(branchless_min(1.5f, 2.5f) == 1.5f);
    assert(branchless_max(1.5f, 2.5f) == 2.5f);

    // Edge: INT_MIN / INT_MAX
    assert(branchless_min(INT_MIN, 0)   == INT_MIN);
    assert(branchless_max(INT_MAX, 0)   == INT_MAX);
    assert(branchless_min(INT_MAX, INT_MIN) == INT_MIN);

    printf("PASS  test_min_max\n");
}

// ---------------------------------------------------------------------------
// Test 4: branchless_abs — integers and floats
// ---------------------------------------------------------------------------
static void test_abs() {
    // int32
    assert(branchless_abs(0)    == 0);
    assert(branchless_abs(5)    == 5);
    assert(branchless_abs(-5)   == 5);
    assert(branchless_abs(INT_MAX)  == INT_MAX);
    // INT_MIN is a known edge case: abs(INT_MIN) overflows in 2's complement.
    // The bit-manipulation formula gives INT_MIN (same bit pattern) — same as
    // most libc abs() implementations. Document and accept this behaviour.
    assert(branchless_abs(INT_MIN) == INT_MIN);

    // int64
    assert(branchless_abs(int64_t{-1'000'000'000LL}) == 1'000'000'000LL);

    // float
    assert(branchless_abs(-3.14f) == 3.14f);
    assert(branchless_abs( 3.14f) == 3.14f);
    assert(branchless_abs( 0.0f)  == 0.0f);

    // double
    assert(branchless_abs(-2.718) == 2.718);

    printf("PASS  test_abs\n");
}

// ---------------------------------------------------------------------------
// Test 5: branchless_clamp
// ---------------------------------------------------------------------------
static void test_clamp() {
    // Below range
    assert(branchless_clamp(-5, 0, 10) ==  0);
    // In range
    assert(branchless_clamp( 5, 0, 10) ==  5);
    // Above range
    assert(branchless_clamp(15, 0, 10) == 10);
    // Equal to bounds
    assert(branchless_clamp( 0, 0, 10) ==  0);
    assert(branchless_clamp(10, 0, 10) == 10);

    // float
    assert(branchless_clamp(-1.0f, 0.0f, 1.0f) == 0.0f);
    assert(branchless_clamp( 0.5f, 0.0f, 1.0f) == 0.5f);
    assert(branchless_clamp( 2.0f, 0.0f, 1.0f) == 1.0f);

    // uint8 (activation clamping pattern)
    assert(branchless_clamp(uint8_t{200}, uint8_t{0}, uint8_t{127}) == 127);
    assert(branchless_clamp(uint8_t{50},  uint8_t{0}, uint8_t{127}) == 50);

    printf("PASS  test_clamp\n");
}

// ---------------------------------------------------------------------------
// Test 6: branchless_relu
// ---------------------------------------------------------------------------
static void test_relu() {
    assert(branchless_relu(-5)   ==  0);
    assert(branchless_relu( 0)   ==  0);
    assert(branchless_relu( 5)   ==  5);
    assert(branchless_relu(-0.5f) == 0.0f);
    assert(branchless_relu( 0.5f) == 0.5f);
    assert(branchless_relu(-1.0)  == 0.0);
    printf("PASS  test_relu\n");
}

// ---------------------------------------------------------------------------
// Test 7: branchless_sign
// ---------------------------------------------------------------------------
static void test_sign() {
    assert(branchless_sign(-5)    == -1);
    assert(branchless_sign( 0)    ==  0);
    assert(branchless_sign( 5)    ==  1);
    assert(branchless_sign(-0.5f) == -1.0f);
    assert(branchless_sign( 0.0f) ==  0.0f);
    assert(branchless_sign( 0.5f) ==  1.0f);
    assert(branchless_sign(INT_MIN) == -1);
    assert(branchless_sign(INT_MAX) ==  1);
    printf("PASS  test_sign\n");
}

// ---------------------------------------------------------------------------
// Test 8: branchless_between
// ---------------------------------------------------------------------------
static void test_between() {
    assert( branchless_between(5,  0, 10));
    assert( branchless_between(0,  0, 10));
    assert( branchless_between(10, 0, 10));
    assert(!branchless_between(-1, 0, 10));
    assert(!branchless_between(11, 0, 10));

    // Unsigned — no negative values possible, but test the boundary
    assert( branchless_between(uint8_t{127}, uint8_t{0}, uint8_t{127}));
    assert(!branchless_between(uint8_t{128}, uint8_t{0}, uint8_t{127}));

    printf("PASS  test_between\n");
}

// ---------------------------------------------------------------------------
// Test 9: correctness on large arrays (exercises the hot path code)
// ---------------------------------------------------------------------------
static void test_array_ops() {
    constexpr int N = 10000;
    int arr[N];
    for (int i = 0; i < N; ++i) arr[i] = i - N / 2;  // -5000..+4999

    int64_t sum_relu = 0, sum_abs = 0, sum_clamp = 0;
    for (int i = 0; i < N; ++i) {
        sum_relu  += branchless_relu(arr[i]);
        sum_abs   += branchless_abs(arr[i]);
        sum_clamp += branchless_clamp(arr[i], -100, 100);
    }

    // relu: sum of 1..4999 = 4999*5000/2
    assert(sum_relu == 4999LL * 5000 / 2);

    // abs: values are -5000..-1,0,1..4999
    // = abs(-5000) + 2*(1+..+4999) = 5000 + 4999*5000 = 5000*5000
    assert(sum_abs == 5000LL * 5000);

    // clamp: -100..-100 (4900 times), then -100..+100, then 100..100
    // Actually: each of -5000..-101 clamps to -100, -100..100 stays, 101..4999 clamps to 100
    // 4900 values at -100 + (-100..100 = 201 values) + 4899 values at 100
    // recompute expected clamp sum
    int64_t recompute = 0;
    for (int i = -5000; i < 5000; ++i)
        recompute += (i < -100 ? -100 : (i > 100 ? 100 : i));
    assert(sum_clamp == recompute);

    printf("PASS  test_array_ops  (relu=%lld  abs=%lld  clamp=%lld)\n",
           static_cast<long long>(sum_relu),
           static_cast<long long>(sum_abs),
           static_cast<long long>(sum_clamp));
}

int main() {
    test_select();
    test_select_bits();
    test_min_max();
    test_abs();
    test_clamp();
    test_relu();
    test_sign();
    test_between();
    test_array_ops();
    printf("\nAll branchless tests passed.\n");
}
