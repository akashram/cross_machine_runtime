#pragma once

// Branchless Primitives
// =========================================================================
//
// WHY BRANCHES ARE EXPENSIVE
// --------------------------
// Modern out-of-order CPUs speculatively execute past conditional branches.
// When the speculation turns out to be wrong (a misprediction), the CPU
// must flush the reorder buffer and refill the pipeline: ~15–20 cycles of
// wasted work on Skylake (40–70 ns at 3 GHz).
//
// Branch misprediction rate depends on data:
//   Sorted data:   ~0%   (branch predictor learns the pattern easily)
//   Skewed data:   ~0%   (e.g. 99% true → predictor always predicts true)
//   Random data:  ~50%   (50/50 coin flip — predictor can't help)
//
// For latency-critical hot paths processing unpredictable data (neural
// network activations, tree traversal, quantization thresholds), a 50%
// misprediction rate at 20 cycles/miss dominates runtime.
//
//
// THE CMOV SOLUTION
// -----------------
// x86 CMOV (Conditional MOVe) instructions execute without branching.
// The CPU always computes the address/value, then moves it or not based on
// the condition — no speculation, no misprediction, no pipeline flush.
//
//   CMOVG  (conditional move if greater)
//   CMOVL  (conditional move if less)
//   CMOVNE (conditional move if not equal)
//   ...
//
// A C++ ternary `cond ? a : b` compiles to CMOV for scalar integer/pointer
// types at -O2. Floating-point uses FCMOV or a similar sequence.
//
// Trade-off: CMOV always computes both branches, then discards one.
//   ✓ Use when: data is unpredictable, computation is cheap (compare, add).
//   ✗ Avoid when: one branch is expensive (function call, cache miss, divide)
//     — always computing the expensive path wastes cycles.
//   ✗ Avoid when: data is predictable — branch predictor is free,
//     cmov adds a data dependency chain.
//
//
// BIT-MANIPULATION ABS
// --------------------
// For signed integers, arithmetic right shift fills with the sign bit:
//   -3 >> 31 = 0xFFFFFFFF  (all ones, i.e. -1 in 2's complement)
//    3 >> 31 = 0x00000000  (all zeros)
//
// Using this mask: abs(v) = (v ^ mask) - mask
//   v=+3: mask=0,  (3 ^ 0) - 0 = 3        ✓
//   v=-3: mask=-1, (-3 ^ -1) - (-1) = 2+1 = 3  ✓
//
// This generates two arithmetic instructions with no branch or cmov.
// Preferred over the cmov approach for abs() — one fewer operation.
//
//
// RELU — NEURAL NETWORK FAST PATH
// --------------------------------
// ReLU (Rectified Linear Unit) is max(0, v). For FP32:
//   Branchy:      if (v < 0.f) v = 0.f;   → conditional branch on sign
//   Branchless:   MAXSS xmm0, [zero]       → single SSE instruction
//
// At scale (millions of activations per forward pass), the branch version
// can cause 40–50% mispredictions on typical weight distributions.
//
// =========================================================================

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cpu_engine {

// =========================================================================
// select(cond, a, b) — cond ? a : b without branching
//
// At -O2: compiles to CMOV for integer/pointer types (verified with godbolt).
// At -O0: may produce a branch — use select_bits() if guaranteed branchless
//         operation is needed at any optimization level.
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_select(bool cond, T a, T b) noexcept {
    return cond ? a : b;
}

// =========================================================================
// select_bits<T>(cond, a, b) — branchless at ANY optimization level
//
// Uses integer bit manipulation to avoid cmov dependency on compiler opts.
// Only defined for 32- and 64-bit integer types. Generates:
//   NEG + AND + ANDN + OR  (4 instructions, no branch, no cmov).
//
// Note: this is almost always slower than cmov at -O2 due to the extra
// instructions. Use only when -O2 is not guaranteed (e.g. embedded builds).
// =========================================================================
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T> &&
                                     (sizeof(T) == 4 || sizeof(T) == 8)>>
[[nodiscard]] inline T branchless_select_bits(bool cond, T a, T b) noexcept {
    using U = std::make_unsigned_t<T>;
    // mask = cond ? ~0 : 0  (no branch: relies on unsigned negation)
    U mask = static_cast<U>(-static_cast<U>(static_cast<U>(cond) & 1u));
    return static_cast<T>((mask & static_cast<U>(a)) | (~mask & static_cast<U>(b)));
}

// =========================================================================
// min / max
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_min(T a, T b) noexcept {
    return branchless_select(a < b, a, b);
}

template<typename T>
[[nodiscard]] inline T branchless_max(T a, T b) noexcept {
    return branchless_select(a > b, a, b);
}

// =========================================================================
// abs — branchless absolute value
//
// For signed integers: arithmetic-shift-based bit trick (2 instructions).
// For floating point: clears the IEEE 754 sign bit (1 AND instruction).
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_abs(T v) noexcept {
    static_assert(std::is_arithmetic_v<T>, "branchless_abs requires arithmetic type");
    if constexpr (std::is_unsigned_v<T>) {
        return v;
    } else if constexpr (std::is_integral_v<T>) {
        // Arithmetic right shift: mask = -1 if negative, 0 if positive/zero.
        // abs(v) = (v ^ mask) - mask
        T mask = v >> (sizeof(T) * 8 - 1);
        return static_cast<T>((v ^ mask) - mask);
    } else if constexpr (std::is_same_v<T, float>) {
        // Clear IEEE 754 sign bit — ANDPS on x86 (single instruction)
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        bits &= 0x7FFF'FFFFu;
        std::memcpy(&v, &bits, 4);
        return v;
    } else if constexpr (std::is_same_v<T, double>) {
        uint64_t bits;
        std::memcpy(&bits, &v, 8);
        bits &= 0x7FFF'FFFF'FFFF'FFFFull;
        std::memcpy(&v, &bits, 8);
        return v;
    } else {
        // Fallback: ternary (compiler generates cmov or similar)
        return v < T{0} ? -v : v;
    }
}

// =========================================================================
// clamp(v, lo, hi) — branchless clamp to [lo, hi]
//
// Two cmov instructions: one for lower bound, one for upper bound.
// Equivalent to: max(lo, min(v, hi))
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_clamp(T v, T lo, T hi) noexcept {
    return branchless_min(branchless_max(v, lo), hi);
}

// =========================================================================
// relu(v) — max(0, v), the most common neural network activation
//
// For floats: MAXSS xmm, [zero]    (1 SSE instruction)
// For ints:   CMP + CMOVG          (2 instructions)
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_relu(T v) noexcept {
    return branchless_max(v, T{0});
}

// =========================================================================
// sign(v) — returns -1, 0, or +1 for the sign of v
//
// (0 < v) and (v < 0) each produce 0 or 1 without branching.
// The subtraction gives -1, 0, or +1 with no branch instruction.
// =========================================================================
template<typename T>
[[nodiscard]] inline T branchless_sign(T v) noexcept {
    return static_cast<T>((T{0} < v) - (v < T{0}));
}

// =========================================================================
// between(v, lo, hi) — returns 1 if lo <= v <= hi, else 0
//
// Branchless range check. Useful in parsers, quantizers, and validators.
// =========================================================================
template<typename T>
[[nodiscard]] inline bool branchless_between(T v, T lo, T hi) noexcept {
    // Unsigned subtraction trick: if v is in [lo, hi], then (v-lo) < (hi-lo+1).
    // This works for both signed and unsigned T because we cast to unsigned.
    using U = std::make_unsigned_t<T>;
    return static_cast<U>(v - lo) <= static_cast<U>(hi - lo);
}

} // namespace cpu_engine
