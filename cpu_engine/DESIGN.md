# CPU Backend Architecture

Status: Phase 2 complete (steps 1–13). Primary development platform: macOS
Intel (Apple clang 14, AVX2, no AVX-512, advisory-only thread affinity).
Numbers marked "(Linux, expected)" are documented predictions — derived
from the roofline model and counter-event semantics — to be validated on
an AVX-512 Linux box (target: AWS c5.2xlarge, Xeon Platinum 8275CL).

This document explains *why* each piece of `cpu_engine/` looks the way it
does, the alternative considered for each decision, and the measurement
that justified the choice. Per-component READMEs (`cpu_engine/*/README.md`)
hold the raw numbers; this doc is the synthesis across all 13 steps.

---

## 1. The one number that explains most of Phase 2: the ridge point

Step 10 (roofline) measured this machine's ridge point — the arithmetic
intensity (FLOP/byte) where compute and bandwidth ceilings cross — at
**3.73 FLOP/byte** (peak 66.8 GFLOPS ÷ peak 17.9 GB/s).

Every kernel in `cpu_engine/` falls into one of two camps relative to that
number, and which camp a kernel is in *determines which optimizations can
possibly help it*:

| Camp | AI range | Kernels | What can move the needle |
|---|---|---|---|
| Bandwidth-bound (AI ≪ 3.73) | 0.08 – 0.5 | dot, matvec, eltwise (add/mul/relu/sigmoid), MLP forward | reduce bytes moved (NT stores, hugepages, fusion, batching) |
| Compute-bound (AI ≫ 3.73) | ~42.7 | matmul (256×256) | reduce instructions / improve ILP (tiling, FMA scheduling, branchless) |

This single fact is why steps 4–6 (NT stores, prefetch, branchless) show
*small or even negative* wins on the vector kernels in isolation — those
kernels are limited by DRAM bytes/second, not by instruction count or
branch prediction, so shaving instructions off a memory-bound loop doesn't
change its wall-clock time. It is also why tiling (step 8) only pays off
for matmul, and only once the matrix exceeds L3 (the 1024×1024 case).

Every design decision below is read against this map: "does this kernel
have headroom to gain from the optimization, or is it already pinned to
the bandwidth ceiling?"

---

## 2. Step 1 — CPU affinity / thread pinning

**Decision:** `ThreadPinner` wraps `pthread_setaffinity_np` on Linux (hard
pin — the kernel will never migrate the thread) and `thread_policy_set` /
`THREAD_AFFINITY_POLICY` on macOS (an *advisory* hint the scheduler may
ignore). `current_cpu()` uses `sched_getcpu()` on Linux and decodes
`TSC_AUX` from `RDTSCP` on macOS, since the XNU kernel writes the logical
CPU index there per-core.

**Alternative considered:** skip macOS support entirely and gate the whole
component behind `#ifdef __linux__`, since affinity is "really" a Linux
hard-pinning concern.

**Why not:** Phase 2 steps 1–6 run on macOS day-to-day; an affinity API that
silently no-ops there would make every subsequent jitter measurement
meaningless without warning. Implementing the macOS advisory path — and
having the benchmark *report* that pinning is advisory-only rather than
hard — keeps the numbers honest: the TSC-jitter benchmark shows pinned ≈
unpinned (~1×) on macOS, which is the *correct* result for an advisory
affinity API, not a bug to chase.

**Measured justification:** TSC inter-sample jitter (stddev) over 100k
samples is the metric. Linux hard-pinning is expected to land in the
10–50 ns range vs. 200–2000 ns unpinned (cache migration + branch
predictor cold-start cost) — this is the number every later latency
measurement implicitly depends on, since an unpinned benchmark thread
injects scheduler noise into every p99/p999 tail.

---

## 3. Step 2 — Hugepage allocator

**Decision:** `HugeRegion` is an RAII wrapper that requests 2 MB pages via
`mmap(MAP_HUGETLB)` on Linux, binds them to a NUMA node with
`mbind(MPOL_BIND)` (composing directly with the affinity from step 1: pin
the thread, then bind its memory to the same node), and silently falls
back to 4 KB pages on macOS (`is_huge()` reports which mode is active).
`prefault()` walks every page once before measurement so fault-handling
cost doesn't contaminate the benchmark.

**Alternative considered:** use `madvise(MADV_HUGEPAGE)` (Linux
transparent hugepages) instead of explicit `MAP_HUGETLB`.

**Why not:** THP is opportunistic — the kernel decides whether and when to
collapse pages, which makes the TLB-pressure measurement non-deterministic
run to run. Explicit `MAP_HUGETLB` either gets a 2 MB page or fails
immediately; the allocator can report which one happened
(`expected_tlb_entries()` quantifies the difference: 65,536 dTLB entries
for a 256 MB region at 4 KB vs. 128 at 2 MB — a 512× reduction), and the
benchmark stays reproducible.

**Measured justification:** sequential cache-line scan of a 256 MB region
at both page sizes, instrumented with `PerfCounters` for `dTLB-load-misses`.
Expected Linux result: 10–40% wall-clock speedup and a drop from ~65,000
to ~128 dTLB misses per pass — directly attributable to the 512× reduction
in resident TLB entries, not to any change in the data access pattern.

---

## 4. Step 3 — OS-level tuning scripts

**Decision:** Five composable shell scripts (`tune_isolcpus`,
`tune_irq_affinity`, `tune_cstate`, `tune_governor`, orchestrated by
`tune_all` / reversed by `restore_all`), each independently toggleable, each
measuring before/after jitter with the same `measure_jitter.sh` harness.

**Alternative considered:** a single monolithic "latency mode" script that
flips every knob at once.

**Why not:** a monolithic script answers "is the machine faster" but not
"which knob bought how much." Splitting them lets `tune_all.sh` report a
per-knob jitter delta, which is the actual deliverable — without it, a
future regression ("why did p999 jump?") has no way to bisect which OS
setting changed. The order matters too: `isolcpus`/`nohz_full` require a
GRUB edit and reboot, so they're sequenced first and flagged as
non-reversible-without-reboot in `restore_all.sh`, while governor/C-state/
IRQ changes are live-reversible.

**Key mechanism documented:** `nohz_full` is not just "fewer interrupts" —
it changes *when* a sleeping thread wakes. Without it, a thread blocked on
a futex wakes on the next 4 ms tick boundary; with it, the wakeup fires at
the exact timer expiry. This is the single biggest lever on `os-wait` tail
latency (see step 13) and the reason `tune_isolcpus.sh` is listed first
despite being the most invasive change.

**Platform note:** Linux-only by construction (`isolcpus`, `/proc/irq/`,
`cpuidle`, `cpufreq` are kernel interfaces with no macOS equivalent); the
component documents *why* rather than stubbing a no-op macOS path, since
there is no advisory analog the way there was for affinity.

---

## 5. Step 4 — Non-temporal store primitives

**Decision:** `nt_store.h` provides raw typed `nt_store<T>()` (no implicit
fence — caller controls batching), plus `nt_memset`/`nt_memcpy` which
choose AVX2 (`_mm256_stream_si256`, 4×32B/iter) when available, fall back
to SSE2, and end with `sfence()`.

**Alternative considered (the one that almost shipped):** make NT stores
the *default* path for all large sequential writes in the runtime — the
textbook advice for "write-only, won't-be-read-again" data.

**Why not — and this is the most important negative result in Phase 2:**
the benchmark measured **regular `memset` at 81.6 GB/s vs. NT `memset` at
28.2 GB/s** on this Skylake chip — NT stores are *2.9× slower* for zero-fill.
The reason is that modern Intel's `rep stosb` (ERMSB — Enhanced REP MOVSB/
STOSB) is itself a non-allocating, hardware-accelerated streaming store
implemented in microcode; it already does what NT stores are for, faster,
because it can use the full width of the store/fill datapath rather than
being bottlenecked by `_mm256_stream_si256`'s per-instruction throughput.
**Lesson recorded in the README:** always benchmark against the compiler/
libc baseline before hand-rolling a "faster" primitive — the textbook
technique can be a *regression* on a specific microarchitecture. NT stores
remain the right tool for non-zero fill, DMA staging buffers, and
`sfence`-based producer/consumer ordering protocols, where ERMSB doesn't
apply — but "use NT stores for large writes" as a blanket rule is wrong on
this hardware.

---

## 6. Step 5 — Software prefetch primitives

**Decision:** `PrefetchHint` enum mapping to `__builtin_prefetch`
locality levels (T0/T1/T2/NTA), plus `prefetch_ahead<T,H>(base, i, dist)` as
the canonical loop helper, and `make_pointer_chase_list()` for constructing
an access pattern the hardware prefetcher cannot predict.

**Alternative considered:** measure prefetch only on a regular strided scan
— the case the hardware prefetcher already handles well.

**Why not:** that would only prove software prefetch is *redundant*, which
is true but not useful. Two adversarial patterns were chosen instead, each
isolating a different prefetch use case:

- **Pointer chase** (hardware prefetcher structurally blind — next address
  is data-dependent): T0 hint gave a 1.2× speedup (74.1 → 62.7 ns/access).
  Modest, because the chain-walk *itself* serializes the dependent loads —
  you cannot prefetch element N+2 until you've loaded element N+1 to find
  its address. This bounds how much software prefetch can ever help linked
  structures.
- **Large-stride scan** (stride = 2 KB, beyond where the hardware
  prefetcher's stride detector locks on): NTA hint gave a 2.1× speedup
  (11.9 → 5.7 ns/access). NTA wins specifically *because* the data is read
  once — it avoids evicting the working set from L2/L3 with data that will
  never be touched again.

**Decision rule extracted and encoded in the header docs:** T0 for data
that will be reused (keep it in L1), NTA for single-pass streaming (don't
pollute the cache), and *don't bother* below ~512-byte strides — the
hardware prefetcher already wins there and software hints just add
instruction overhead.

---

## 7. Step 6 — Branchless primitives

**Decision:** `select`/`min`/`max`/`abs`/`clamp`/`relu`/`sign`/`between`,
each implemented two ways depending on whether the type allows a cheaper
trick: integer `abs` uses the arithmetic-shift mask trick (2 instructions,
no branch, no `cmov`), float `abs` clears the IEEE-754 sign bit (1
instruction), and the general case is a `cmov`-based ternary.
`select_bits` specifically uses unsigned negation to mask, because that
form is guaranteed branchless at `-O0` — the ternary forms depend on the
optimizer recognizing the pattern.

**Alternative considered:** write everything as `cond ? a : b` and trust
`-O2` to emit `cmov`.

**Why not purely that:** Apple clang *does* already emit `cmov` for the
simple ternaries at `-O2` (confirmed — benchmark shows 1.0× vs. the
hand-written version on macOS, i.e., no daylight between them). But that
guarantee evaporates at `-O0`/`-Og` (debug builds, where branch-prediction-
sensitive tests need to *demonstrate* the bug they're guarding against) and
for the bit-manipulation forms where the compiler has no canonical pattern
to recognize. The header keeps both: trust the optimizer where it's proven
reliable, hand-roll where the guarantee doesn't hold across optimization
levels.

**Measured justification:** branch-miss rate is the ground truth, not
instruction count — a branchy version that predicts perfectly costs nothing
extra; a branchless version of an *already-predictable* branch can lose by
removing a free `cmov`-eligible pattern. The Linux counter run
(`perf stat -e branch-misses,branches`) on randomized data is the
controlled experiment: branch-misses goes from ~50% (coin-flip — maximally
unpredictable) to 0% (no branch exists to mispredict). That 50%→0% delta,
not a synthetic instruction count, is the number that would justify
shipping the branchless form in a hot loop with data-dependent branches.

---

## 8. Step 7 — AVX-512 kernel library (three-tier design)

**Decision:** every kernel is implemented three times:

1. **Tier 1 (scalar)** — reference implementation, always builds, used as
   the correctness oracle and the "no vectorization" baseline.
2. **Tier 2 (auto-vectorized)** — pragma-annotated loops; the compiler
   picks the widest ISA it has (AVX2 here, AVX-512 on a Linux/avx512
   build).
3. **Tier 3 (explicit AVX-512 intrinsics)** — `_mm512_*`, guarded by
   `#ifdef __AVX512F__`, compiled only with the new `avx512` CMake preset.

**Alternative considered:** write Tier 3 only, gated entirely behind
`__AVX512F__`, and accept that the library doesn't build or run on macOS
until the Linux box is available.

**Why not:** that would block steps 8–13 (tiling, inference, roofline,
counters, PGO, busy-poll) on cloud access for two months, and — more
importantly — it would throw away the Tier 1 vs. Tier 2 comparison, which
is itself a load-bearing measurement: **Tier 2 (AVX2 auto-vec) beat Tier 1
(scalar) by 13–16× on this machine** (`dot_f32`: 1.4 → 19.3 GFLOPS;
`matvec`: 2.0 → 32.2 GFLOPS). That comparison is the evidence that
"hand-write AVX-512 intrinsics" is worth doing *at all* — if the compiler's
auto-vectorizer already captured most of the win, Tier 3 would be a
maintenance cost for a small delta. The three-tier structure makes that
argument measurable rather than assumed, and keeps the whole library
buildable and testable on the primary dev machine.

**Numerically interesting choice — `eltwise_sigmoid_f32`:** implemented as
the fast rational approximation `0.5·x/(1+|x|) + 0.5` via
`_mm512_abs_ps` + `_mm512_div_ps` + FMA, rather than a true `exp`-based
sigmoid. This sidesteps a dependency on SVML (Intel's vectorized libm,
not guaranteed present) at the cost of <0.5% error for `|x| < 5` — an
acceptable trade for an inference-engine activation function (step 9),
where the input range is bounded by preceding layers' weight scales.

---

## 9. Step 8 — Cache-aware tiling

**Decision:** `matmul_tiled_f32` blocks the classic `(i,k,j)` triple loop
into `(i₀,k₀,j₀)` outer tile loops over a runtime-configurable square tile
size `T`, chosen so the working set `3T² × 4 bytes` (A-tile + B-tile +
C-accumulator) fits a target cache tier.

**Alternative considered:** pick one "good" tile size from the working-set
formula (e.g., T=64 → 48 KB, fits comfortably in a 256 KB L2) and ship that
as the only configuration.

**Why not:** the measured data showed the "obviously correct" answer is
matrix-size-dependent and *non-monotonic* — and a single hardcoded T would
have hidden a real bug. At 256×256, **T=48 measured 4–5× slower than its
neighbors** (6.1 GFLOPS vs. 19–24 GFLOPS for T=32/64/96–256). This isn't
noise: it's a cache *conflict-miss* artifact — a 256-element row stride
(1024 bytes = 16 cache lines) combined with a 192-byte tile-row stride
(T=48 × 4 bytes) happens to map repeatedly onto the same L1 sets, evicting
live data mid-computation. Power-of-two tile widths sidestep this because
they can't produce that kind of periodic set collision. **The only way to
find this was to sweep T and plot the curve** — a single hardcoded choice
would have shipped a 4–5× landmine that only manifests for specific
matrix-dimension/tile-size combinations.

The second non-obvious result: **tiling *loses* to the naive loop at
256×256 and 512×512** (naive 27.0 / 23.1 GFLOPS vs. best tiled ~21–24 / 21.4
GFLOPS) because this machine's L3 is large enough (~8 MB) that a 3 MB
working set already fits — tiling overhead (outer-loop bookkeeping, worse
instruction-cache locality) outweighs a cache benefit that barely exists.
**Tiling only wins once the matrix exceeds L3** — at 1024×1024 (12 MB),
T=512 gives a real 1.3× speedup (16.2 → 20.8 GFLOPS) because the 3 MB tile
now fits where the naive version thrashes DRAM.

**Decision rule encoded:** tile size is a *runtime* parameter, not a
compile-time constant, specifically because the crossover point is a
function of `(matrix size, L3 size, tile-stride/cache-set geometry)` — all
of which vary across the macOS dev machine and the eventual Linux/AVX-512
deployment target (smaller per-core L3 → earlier crossover, per the
roofline note in the README).

---

## 10. Step 9 — CPU MLP inference engine

**Decision:** `MlpInferenceEngine` pre-allocates everything at construction
— weights, biases, and exactly two "ping-pong" scratch buffers sized to
`max(all layer dims)` — and `forward()` swaps source/destination pointers
each layer. Bias-add and activation are fused into a single in-place pass
over the destination buffer rather than using a third temporary.

**Alternative considered:** allocate a fresh output buffer per layer (the
natural way to write `forward()` if you're not thinking about the
allocator), or a separate bias buffer + separate activation buffer (the
natural way to write it if you *are* composing existing kernels naively).

**Why not:** Phase 2's non-negotiable is "no heap on the hot path" — and
that's not a style preference, it's a latency guarantee. A per-layer `new`
would put a (potentially lock-contending, definitely page-faulting-on-first-
touch) allocator call on every inference. The ping-pong design makes the
allocation *count* — not just the allocation *cost* — independent of
network depth: two buffers serve a 3-layer or a 30-layer network
identically. **Verified, not asserted:** `mlp_test.cpp` overrides all four
global `operator new`/`delete` forms and counts calls across 1,000
consecutive `forward()` invocations on two differently-shaped networks —
both report exactly **0 allocations**. This is the kind of guarantee that
degrades silently (one refactor reintroduces a `std::vector` and the
guarantee is gone with no compile error) — hence the standing regression
test rather than a one-time check.

**Measured justification — why the "small" network is the sweet spot:**
the engine was benchmarked at three sizes, and the result inverts the naive
intuition that "bigger network = more work = lower throughput, smoothly":

| Network | Weight matrix size | GFLOPS | Why |
|---|---|---|---|
| Tiny (64→…→32) | ≤ 64 KB total | 10.9 | fits L1 — fast, but few total FLOPs to amortize fixed overhead |
| **Small (256→…→128)** | **~256 KB** | **17.6** | **fits L2 — matvec stays compute-bound, enough FLOPs to amortize overhead: the actual sweet spot** |
| Large (1024→…→128) | first layer = 8 MB | 10.8 | exceeds L3 — matvec becomes bandwidth-bound, ~17.9 GB/s DRAM ceiling dominates |

Tying back to §1: every MLP configuration measured at AI ≈ 0.49 FLOP/byte
— roughly 8× below the 3.73 ridge point — so **all of them are
bandwidth-bound by the roofline model**, and the GFLOPS differences above
are really just "how much of the working set fits in which cache tier,"
not differences in compute efficiency. The one lever that *would* cross
the ridge — batching, which raises AI to ≈ `0.49 × B` by reusing the same
weight matrix across B inputs — is named explicitly as the next
architectural step if MLP throughput becomes the bottleneck in a larger
pipeline.

---

## 11. Step 10 — Roofline model

**Decision:** measure the machine's *actual* ceilings rather than reading
them off a spec sheet — peak FLOPS via 8 independent AVX2 FMA accumulator
chains over 4M iterations (66.8 GFLOPS), peak bandwidth via STREAM Triad
over a 192 MB working set, best-of-10 (17.9 GB/s) — then classify every
kernel in the library against the resulting ridge point (3.73 FLOP/byte).

**Alternative considered:** use vendor-quoted peak numbers (Intel ARK lists
turbo-boost FLOPS; DDR4-2666 dual-channel nominal bandwidth is ~42 GB/s).

**Why not:** vendor numbers describe a configuration this machine isn't
running — turbo boost is thermally throttled under sustained AVX load, and
measured single-channel-effective bandwidth here is 17.9 GB/s, not 42.
Using spec numbers would have put *every* kernel's "% of ceiling" in the
20–40% range and made the roofline chart useless for distinguishing "this
kernel has room to improve" from "this kernel is already at the practical
limit of this hardware." Measuring the actual ceiling is what makes
"matmul tiled is at 67% of peak, naive is at 48%" a meaningful, actionable
gap rather than noise against an unreachable number.

**The finding that reframes the whole library:** of the eleven kernels
sampled, **ten are bandwidth-bound** (AI 0.08–0.5, all far below the 3.73
ridge) and **only matmul is compute-bound** (AI ≈ 42.7). This means most of
the "make it faster" levers available in Phase 2 — branchless code,
prefetch, NT stores, AVX-512 width — *cannot* move the needle on ten of
eleven kernels, because their ceiling is `bandwidth × AI`, a number that
doesn't change when you make the inner loop execute in fewer cycles. The
only things that *can* help those ten are: move fewer bytes (fusion,
narrower dtypes, NT stores for write-only paths), or raise AI (batching,
weight reuse, blocking that increases reuse-per-byte-loaded). This
single chart is the reason step 8 (tiling) targeted matmul specifically,
and why step 9 (MLP) names batching — not a faster kernel — as its next
lever.

---

## 12. Step 11 — Hardware perf counter deep dive

**Decision:** extend `foundation/perf/PerfCounters` (built in Phase 1 step
18) with `ExtendedPerfCounters`, adding L1D/LLC access-and-miss events via
`PERF_TYPE_HW_CACHE`, and run every kernel at two working-set sizes
(in-cache vs. DRAM-resident) to produce a per-kernel IPC / cache-miss /
branch-miss table.

**Implementation decision worth recording:** the eight events are opened as
**independent file descriptors**, not a single `PERF_FORMAT_GROUP`, because
mixing `PERF_TYPE_HARDWARE` and `PERF_TYPE_HW_CACHE` in one group is not
universally supported across kernel versions — grouping would have made
the tool less portable for a marginal convenience. Since the PMU has only
~4 programmable counters and 8 events are requested, the kernel
multiplexes; `PERF_FORMAT_TOTAL_TIME_ENABLED`/`RUNNING` is read back and
used to scale each raw count to its estimated true value. This is
documented explicitly so that anyone reading the counter output knows the
numbers are *estimates under multiplexing*, not exact counts — a subtlety
that would otherwise produce confusing "why don't these add up" questions.

**Why this step exists at all (rather than stopping at the roofline):** the
roofline model classifies kernels as bandwidth- or compute-bound from
*first principles* (AI vs. ridge point) — it's a prediction. This step is
the thing that would *falsify* that prediction if the hardware behaved
differently than the model assumes. The README encodes this as a literal
table of IPC/L3-miss-rate predictions per kernel ("expect IPC < 1.5 and L3
miss > 60% for DRAM-resident dot/matvec/relu/add because they're
bandwidth-bound; expect IPC 3–4 and L3 miss < 5% for matmul because it's
compute-bound") to be checked against real counter data on Linux. It also
names the specific anomaly to look for — an L1-miss-rate spike at T=48,
the tiling conflict-miss artifact from step 8 — turning a "huh, that's
weird" result from step 8 into a falsifiable hypothesis for step 11 to
confirm or refute with hard counter evidence.

**Why `perf_event_open` is unavailable on macOS** is also documented in the
header rather than left as a silent `false`: XNU does not expose the Linux
PMU interface, so `available()` returns false and all fields read zero.
Code that depends on counter data (e.g., regression gates) must check
`available()` and degrade to timing-only — which is exactly what this
benchmark does, reporting a `ns/call` table on macOS and noting that the
IPC/miss-rate columns require the Linux run.

---

## 13. Step 12 — Profile-guided optimization (PGO)

**Decision:** ship the full instrument → train → merge → recompile → compare
workflow (`run_pgo.sh`) and *publish the results even though one of them is
a 52% regression*.

**Why publish a regression instead of re-running until it looks good:**
because the regression is the most instructive result PGO produced, and
silently omitting it would leave a future reader to rediscover — the hard
way, in production — that PGO is not a strictly-positive lever. The
analysis nails down *why* it regressed:

1. **Profile/binary mismatch.** The `pgo-use` build emitted `1 has
   mismatched data that will be ignored` — `matmul_tiled_f32`'s profile was
   discarded after an unrelated source change (a pragma removal in
   `mlp.h`) shifted the binary layout enough to invalidate it. With its
   profile gone, the compiler fell back to *more conservative* heuristics
   than it would use with no PGO data at all — a "worse than doing nothing"
   outcome that's specific to instrumentation-based PGO's exact-match
   requirement.
2. **SIMD inner loops have nothing for PGO to learn.** Matmul's hot loop is
   a triple-nested loop with no unpredictable branches; the compiler's
   default vectorization width / unroll factor / pipelining choices were
   already near-optimal. Profile data about loop trip counts perturbed the
   unroll factor into one that interacts worse with the hardware pipeline.

**The decision rule this produces** (documented as a lessons table in the
README) is the actually-useful deliverable: PGO helps **branchy dispatch**
(the MLP's per-layer `switch(activation)` — correct branch ordering, cold-
path layout), is **neutral** on **pure streaming** kernels (dot, eltwise —
nothing to predict, already vectorized), and is **neutral-to-negative** on
**compute-bound SIMD** (matmul — can disrupt vectorization decisions that
were already correct). And the meta-lesson: **PGO is not free** — the
profile must exactly match the binary, which means in any real deployment
PGO has to be a locked CI step (build → profile → recompile, same source
tree, same compiler flags), not a one-time manual tune-up that silently
goes stale on the next commit.

The MLP numbers themselves were marked "noisy, inconclusive" rather than
forced into a conclusion — three independent macOS-specific noise sources
are named (advisory-only pinning lets the OS migrate mid-measurement,
turbo-boost clock variation between cold/warm runs, and a measurement
window of only 3–4 ms total). The Linux re-run with hard affinity, locked
clocks, and `nohz_full` is specified as the controlled experiment that
would actually answer the MLP question — the honest move was to say "this
measurement isn't trustworthy enough to draw a conclusion from" rather than
report a number that *looks* precise.

---

## 14. Step 13 — Busy-poll vs. OS-wait

**Decision:** measure both consumer strategies — spin-on-`pop()` with `PAUSE`
hints vs. block-on-`condition_variable` — across five inter-arrival rates
(tight-loop, 1 µs, 10 µs, 100 µs, 1 ms), reporting p50/p99/p999 for each,
rather than a single "average latency" number.

**Alternative considered:** report mean latency and a single recommended
crossover threshold ("spin below X µs, block above X µs").

**Why not:** the tail is the entire point of this measurement — a
producer/consumer scheduling decision is a latency-SLA decision, and SLAs
are stated in p99/p999, not means. The data shows *why* a single threshold
would be misleading: busy-poll's p50 is rock-stable at 122–139 ns across
every realistic inter-arrival rate (1 µs–1 ms) — that's the
**cache-coherence round-trip** through the SPSC queue (producer
store-release → MESIF transfer across the shared L3 → consumer
load-acquire), measured at ~50–200 ns and landing right where the protocol
predicts. OS-wait's p50 is *also* essentially flat at 3.7–4.3 µs — the
**minimum scheduler wakeup latency**, a fixed floor independent of how
often messages arrive (until 1 ms inter-arrival, where deeper C-states push
it to 38 µs). **Busy-poll is 30–300× lower p50 latency at every rate
tested, and the crossover never appears within the tested range** — a
single-number "use polling below X" recommendation would have implied a
crossover that the data says doesn't exist in any latency-sensitive regime.

**The actually-correct decision rule, extracted from the data:** the real
crossover isn't latency, it's **CPU utilization** — busy-poll burns a full
core regardless of message rate; OS-wait yields the core between messages.
The choice is therefore "do you have a spare core to dedicate" (→ poll, get
the 30–300× latency win) vs. "is this thread sharing a core with other
work" (→ block, accept the µs-scale wakeup floor). This reframes the
question from a latency threshold (which the data shows doesn't move) to a
resource-allocation question (which does), and is the version of the
finding that's actually actionable when designing the scheduler in
`foundation/`.

**Tight-loop anomaly flagged, not hidden:** busy-poll's tight-loop p50
(343 µs) is *worse* than its 1 µs row (122 ns) — counterintuitive, since
"more messages" should mean "less idle spinning." The README calls this out
explicitly as an anomaly to be re-measured on Linux with hard pinning,
rather than averaging it away or omitting the tight-loop row — almost
certainly a measurement artifact of saturating the SPSC queue faster than
the consumer can drain it on an advisory-pinned macOS thread, but that's a
hypothesis to verify, not a fact to assert.

---

## 15. Cross-cutting lessons that should inform Phase 3+

1. **Measure the machine before measuring the kernel.** Steps 4 (NT vs.
   ERMSB memset), 10 (measured vs. vendor-quoted ceilings), and 12 (PGO
   regression from profile mismatch) all hinge on a baseline that looked
   "obviously slower" on paper but wasn't, on this hardware, in this
   compiler. The textbook technique is a hypothesis, not a guarantee —
   every primitive in this library is benchmarked against the
   straightforward alternative before being adopted.

2. **Classify before optimizing.** The roofline model (step 10) is not
   decoration — it's the gate that determines whether an optimization *can*
   work before time is spent attempting it. Ten of eleven kernels are
   bandwidth-bound; no amount of instruction-level cleverness changes their
   ceiling. This should be the first question asked of any new GPU kernel
   in Phase 3 too: is it compute-bound (where warp-level/Tensor-Core
   tricks matter) or bandwidth-bound (where coalescing and occupancy
   matter more than raw throughput)?

3. **Publish the negative result.** The NT-store regression, the T=48
   tiling cliff, and the PGO matmul regression are the three most useful
   findings in Phase 2 — each one encodes a decision rule ("don't do X
   when Y") that a clean success story wouldn't have produced. Phase 3's
   GPU work should keep this discipline: a kernel that doesn't beat its
   baseline is a finding, not a failure to hide.

4. **macOS-vs-Linux divergence is itself data, not noise.** Affinity
   (advisory vs. hard), hugepages (none vs. `MAP_HUGETLB`), and
   `perf_event_open` (absent vs. present) aren't just "things that don't
   work on the dev machine" — documenting *what specifically differs and
   why* (XNU has no PMU interface; Mach has no hard-pinning API) is what
   makes the Linux re-run a confirmation of a stated prediction rather than
   a fresh investigation from zero. Every component that diverges
   platform-to-platform states its Linux-expected numbers in advance.

---

## 16. Open items for the Linux validation pass

These are the predictions made throughout Phase 2 that are currently
unverified on real AVX-512 hardware (target: AWS c5.2xlarge):

- Hard-pin jitter: 10–50 ns stddev (vs. ~1× ratio measured on macOS advisory pinning)
- Hugepage TLB miss reduction: 65,536 → 128 dTLB entries, 10–40% wall-clock win
- AVX-512 absolute throughput: ~192 GFLOPS peak (vs. 66.8 measured AVX2), ridge point ~4 FLOP/byte (materially unchanged — all current bandwidth-bound classifications should hold)
- Counter table: IPC < 1.5 / L3-miss > 50% for bandwidth-bound kernels, IPC 3–4 / L3-miss < 5% for matmul; **T=48 L1-miss spike** as the specific falsifiable claim from the tiling anomaly
- PGO on MLP: 5–15% win expected with hard affinity + locked clocks + `nohz_full` (vs. "noisy, inconclusive" on macOS)
- Busy-poll tight-loop anomaly (343 µs p50, worse than the 1 µs row): re-measure with hard pinning to determine whether it's a scheduling artifact or a real saturation effect
- `nohz_full` impact on OS-wait floor: expected to push the 3.7–4.3 µs macOS floor down to 2–5 µs (or below 2 µs combined with `SCHED_FIFO`)
