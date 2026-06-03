#pragma once

// Minimal Property-Based Testing Framework
// =========================================================================
//
// Property-based testing (QuickCheck, Hypothesis, rapidcheck) generates
// random inputs, runs them through a predicate that should always hold, and
// — critically — SHRINKS any counterexample to the smallest failing case
// before reporting it.
//
// Without shrinking you get: "FAIL with input vector<int>{42, -7, 0, 99, ...
//   (127 more elements)}".
// With shrinking you get:    "FAIL with input vector<int>{-1}".
//
//
// This header provides:
//
//   Rng             — splitmix64 PRNG, reproducible by seed
//   Gen<T>          — lazy generator: (Rng&, size_t complexity) → T
//   shrink(T)       — sequence of "simpler" T values for counterexample reduction
//   check(...)      — run a property N times; shrink and report on failure
//
//
// Shrinking algorithm
// -------------------
// On finding a failing input V:
//   loop:
//     for each candidate C in shrink(V):
//       if property(C) fails:
//         V = C; restart loop     // found a smaller counterexample
//     break                       // no smaller counterexample found
//   report V as the minimal failing case
//
// This greedy search is O(shrink_candidates * retries) per shrink step.
// It doesn't find the GLOBAL minimum but usually finds a very small one.
//
//
// Why splitmix64?
// ---------------
// Splitmix64 passes BigCrush (the strictest statistical RNG test).  It is
// two 64-bit multiplies + three XOR-shifts per output — faster than Mersenne
// Twister and far simpler.  The seed fully determines the sequence, so any
// failing case is reproducible: re-run with the reported seed.
//
//
// Usage
// -----
//   // Property: reversing a vector twice gives back the original
//   auto result = proptest::check("reverse_involution",
//       proptest::gen_vector(proptest::gen_int(-100, 100)),
//       [](const std::vector<int>& v) {
//           auto w = v; std::reverse(w.begin(), w.end());
//           std::reverse(w.begin(), w.end());
//           return w == v;
//       });
//   if (!result.passed) { ... }
//
// =========================================================================

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace proptest {

// =========================================================================
// Rng — splitmix64
// =========================================================================
class Rng {
public:
    explicit Rng(uint64_t seed = 0x853c49e6748fea9bULL) noexcept
        : state_(seed) {}

    uint64_t next() noexcept {
        state_ += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform integer in [lo, hi] (inclusive).
    int64_t next_int(int64_t lo, int64_t hi) noexcept {
        assert(lo <= hi);
        uint64_t range = static_cast<uint64_t>(hi - lo) + 1;
        if (range == 0) return lo;  // full range: overflow
        return lo + static_cast<int64_t>(next() % range);
    }

    std::size_t next_size(std::size_t lo, std::size_t hi) noexcept {
        assert(lo <= hi);
        std::size_t range = hi - lo + 1;
        return lo + static_cast<std::size_t>(next() % range);
    }

    bool next_bool() noexcept { return (next() & 1) != 0; }

    uint64_t seed() const noexcept { return state_; }

private:
    uint64_t state_;
};

// =========================================================================
// Gen<T> — a generator
// =========================================================================
// `complexity` (0..max_complexity) controls how large/complex the generated
// value is.  Low complexity → small values (easier to find bugs quickly).
// High complexity → larger values (stress-test corner cases).

template<typename T>
struct Gen {
    std::function<T(Rng&, std::size_t complexity)> fn;

    T operator()(Rng& rng, std::size_t complexity) const {
        return fn(rng, complexity);
    }

    template<typename F>
    auto map(F f) const {
        using U = std::invoke_result_t<F, T>;
        auto fn_copy = fn;
        return Gen<U>{[fn_copy, f](Rng& rng, std::size_t c) -> U {
            return f(fn_copy(rng, c));
        }};
    }
};

// =========================================================================
// Built-in generators
// =========================================================================

inline Gen<int> gen_int(int lo = -1000, int hi = 1000) {
    return Gen<int>{[lo, hi](Rng& rng, std::size_t /*c*/) -> int {
        return static_cast<int>(rng.next_int(lo, hi));
    }};
}

inline Gen<int> gen_int_biased() {
    // Mix of edge-case values (0, ±1, limits) and random values.
    return Gen<int>{[](Rng& rng, std::size_t c) -> int {
        int range = static_cast<int>(c) + 1;
        switch (rng.next() % 8) {
            case 0: return 0;
            case 1: return 1;
            case 2: return -1;
            case 3: return range;
            case 4: return -range;
            default: return static_cast<int>(rng.next_int(-range * 4, range * 4));
        }
    }};
}

inline Gen<uint64_t> gen_uint64() {
    return Gen<uint64_t>{[](Rng& rng, std::size_t) -> uint64_t {
        return rng.next();
    }};
}

inline Gen<bool> gen_bool() {
    return Gen<bool>{[](Rng& rng, std::size_t) -> bool {
        return rng.next_bool();
    }};
}

inline Gen<std::size_t> gen_size(std::size_t lo = 0, std::size_t max_hi = 50) {
    return Gen<std::size_t>{[lo, max_hi](Rng& rng, std::size_t c) -> std::size_t {
        std::size_t hi = lo + std::min(max_hi, c + 1);
        return rng.next_size(lo, hi);
    }};
}

template<typename T>
Gen<std::vector<T>> gen_vector(Gen<T> elem, std::size_t max_size = 30) {
    return Gen<std::vector<T>>{
        [elem, max_size](Rng& rng, std::size_t c) -> std::vector<T> {
            std::size_t n = rng.next_size(0, std::min(max_size, c + 1));
            std::vector<T> v;
            v.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                v.push_back(elem(rng, c));
            return v;
        }};
}

// Generates a non-empty vector.
template<typename T>
Gen<std::vector<T>> gen_nonempty_vector(Gen<T> elem, std::size_t max_size = 30) {
    return Gen<std::vector<T>>{
        [elem, max_size](Rng& rng, std::size_t c) -> std::vector<T> {
            std::size_t n = rng.next_size(1, std::max(std::size_t{1},
                                                       std::min(max_size, c + 1)));
            std::vector<T> v;
            v.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                v.push_back(elem(rng, c));
            return v;
        }};
}

// =========================================================================
// Shrinking
// =========================================================================

template<typename T>
std::vector<T> shrink(const T&) { return {}; }  // default: no shrinking

template<>
inline std::vector<int> shrink(const int& n) {
    std::vector<int> out;
    if (n != 0)       out.push_back(0);
    if (n < 0)        out.push_back(-n);
    if (n >  1)       out.push_back(n / 2);
    if (n < -1)       out.push_back(n / 2);
    if (n >  2)       out.push_back(n - 1);
    if (n < -2)       out.push_back(n + 1);
    // Remove duplicates and values equal to n
    out.erase(std::remove(out.begin(), out.end(), n), out.end());
    return out;
}

template<>
inline std::vector<std::size_t> shrink(const std::size_t& n) {
    std::vector<std::size_t> out;
    if (n > 0) out.push_back(0);
    if (n > 1) out.push_back(n / 2);
    if (n > 1) out.push_back(n - 1);
    return out;
}

template<typename T>
std::vector<std::vector<T>> shrink(const std::vector<T>& v) {
    std::vector<std::vector<T>> out;
    if (!v.empty()) out.push_back({});          // try empty
    if (v.size() > 1)
        out.push_back({v.begin(), v.begin() + v.size() / 2});  // try first half

    // Remove one element at a time (up to 8 removals to keep shrink fast)
    std::size_t step = std::max(std::size_t{1}, v.size() / 8);
    for (std::size_t i = 0; i < v.size(); i += step) {
        std::vector<T> w;
        w.reserve(v.size() - 1);
        for (std::size_t j = 0; j < v.size(); ++j)
            if (j != i) w.push_back(v[j]);
        out.push_back(std::move(w));
    }

    // Shrink each element (up to 5 elements to keep shrink fast)
    std::size_t elem_step = std::max(std::size_t{1}, v.size() / 5);
    for (std::size_t i = 0; i < v.size(); i += elem_step) {
        for (const auto& smaller : shrink(v[i])) {
            auto w = v;
            w[i] = smaller;
            out.push_back(std::move(w));
        }
    }
    return out;
}

// =========================================================================
// CheckResult and check()
// =========================================================================

struct CheckResult {
    bool        passed{true};
    int         tests_run{0};
    uint64_t    seed{0};           // seed used (for reproduction)
    int         fail_test{-1};     // which test number failed (-1 = none)
    std::string failure_info;      // human-readable, set by check()
};

inline void print_result(const CheckResult& r, const char* name) {
    if (r.passed) {
        std::printf("OK    %-44s  %d tests\n", name, r.tests_run);
    } else {
        std::printf("FAIL  %-44s  test %d (seed=%llu)\n",
                    name, r.fail_test,
                    static_cast<unsigned long long>(r.seed));
        if (!r.failure_info.empty())
            std::printf("      %s\n", r.failure_info.c_str());
    }
}

// Run `num_tests` trials of `property(gen(rng, complexity))`.
// On failure, greedily shrink the counterexample and report it.
// `to_str` converts a counterexample to a printable string.
template<typename T, typename Property, typename ToStr>
CheckResult check(const char* name,
                  Gen<T> gen,
                  Property property,
                  ToStr to_str,
                  int num_tests = 200,
                  uint64_t seed = 42,
                  std::size_t max_complexity = 50)
{
    CheckResult result;
    result.seed = seed;
    Rng rng(seed);

    for (int i = 0; i < num_tests; ++i) {
        std::size_t c = static_cast<std::size_t>(i) * max_complexity /
                        static_cast<std::size_t>(num_tests);
        T val = gen(rng, c);

        if (!property(val)) {
            result.passed   = false;
            result.tests_run = i + 1;
            result.fail_test = i;

            // Greedy shrink
            for (int round = 0; round < 100; ++round) {
                bool got_smaller = false;
                for (auto& candidate : shrink(val)) {
                    if (!property(candidate)) {
                        val = std::move(candidate);
                        got_smaller = true;
                        break;
                    }
                }
                if (!got_smaller) break;
            }

            result.failure_info = "counterexample: " + to_str(val);
            print_result(result, name);
            return result;
        }
        result.tests_run = i + 1;
    }

    print_result(result, name);
    return result;
}

// Convenience: no to_str (failure info omitted).
template<typename T, typename Property>
CheckResult check(const char* name, Gen<T> gen, Property property,
                  int num_tests = 200, uint64_t seed = 42) {
    return check(name, gen, property,
                 [](const T&) { return std::string("(omitted)"); },
                 num_tests, seed);
}

// Convenience: int property with to_str.
inline CheckResult check_int(const char* name,
                              Gen<int> gen,
                              std::function<bool(int)> prop,
                              int num_tests = 200) {
    return check(name, gen, prop,
                 [](int v) { return std::to_string(v); },
                 num_tests);
}

inline CheckResult check_intvec(const char* name,
                                 Gen<std::vector<int>> gen,
                                 std::function<bool(const std::vector<int>&)> prop,
                                 int num_tests = 200) {
    return check(name, gen, prop,
                 [](const std::vector<int>& v) {
                     std::string s = "[";
                     for (std::size_t i = 0; i < v.size() && i < 8; ++i)
                         s += (i ? "," : "") + std::to_string(v[i]);
                     if (v.size() > 8) s += ",...";
                     s += "]";
                     return s;
                 },
                 num_tests);
}

} // namespace proptest
