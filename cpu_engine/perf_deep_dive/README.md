# Step 11 — Hardware perf counter deep dive

## What was built

`perf_deep_dive.h` — `ExtendedPerfCounters`: a superset of `foundation/perf/PerfCounters` that adds L1D and LLC (L3) cache-access/miss events via `PERF_TYPE_HW_CACHE`.

`bench/perf_counters_bench.cpp` — runs every cpu_engine kernel at two data sizes (in-cache and DRAM-resident), reporting ns/call plus IPC, L1 miss %, L2 miss %, L3 miss %, and branch miss % per-iteration.

## Events measured

| Event | perf_event type | config | Meaning |
|---|---|---|---|
| instructions | `PERF_TYPE_HARDWARE` | `PERF_COUNT_HW_INSTRUCTIONS` | retired instructions |
| cycles | `PERF_TYPE_HARDWARE` | `PERF_COUNT_HW_CPU_CYCLES` | elapsed CPU cycles |
| L1D accesses | `PERF_TYPE_HW_CACHE` | `L1D \| READ \| ACCESS` | total L1D load traffic |
| L1D misses | `PERF_TYPE_HW_CACHE` | `L1D \| READ \| MISS` | loads that missed L1 (→ L2) |
| LLC accesses | `PERF_TYPE_HW_CACHE` | `LL \| READ \| ACCESS` | loads that reached L3 (= L2 misses) |
| LLC misses | `PERF_TYPE_HW_CACHE` | `LL \| READ \| MISS` | loads that went to DRAM |
| branches | `PERF_TYPE_HARDWARE` | `PERF_COUNT_HW_BRANCH_INSTRUCTIONS` | branch count |
| branch misses | `PERF_TYPE_HARDWARE` | `PERF_COUNT_HW_BRANCH_MISSES` | mispredictions |

Because the hardware PMU has ~4 programmable counters and we open 8 events, the kernel multiplexes (each counter runs ~50% of the time). `PERF_FORMAT_TOTAL_TIME_ENABLED / RUNNING` is used to scale each value proportionally.

## Implementation notes

Events are opened as **independent** file descriptors (not a PERF_FORMAT_GROUP) to avoid mixing `PERF_TYPE_HARDWARE` and `PERF_TYPE_HW_CACHE` in one group, which isn't universally supported across kernel versions. We issue separate `PERF_EVENT_IOC_ENABLE` / `DISABLE` ioctls to all fds to start/stop them together.

On macOS: `perf_event_open()` is unavailable. `available()` returns false; all counter fields are 0.

## Measured results (macOS Intel — timing only)

| Kernel | ns/call |
|---|---|
| dot_f32 8KB (L1) | 151 ns |
| dot_f32 64MB (DRAM) | 4 385 293 ns |
| matvec 128×128 (L2) | 980 ns |
| matvec 1024×1024 (L3/DRAM) | 169 008 ns |
| relu 16KB (L1) | 166 ns |
| relu 64MB (DRAM) | 3 625 145 ns |
| add 192MB (DRAM) | 6 236 276 ns |
| matmul_naive 256×256 | 1 104 700 ns |
| matmul_tiled T=64 256×256 | 899 868 ns |
| matmul_tiled T=32 256×256 | 783 943 ns |
| mlp_fwd tiny (L1/L2) | 1 802 ns |
| mlp_fwd small (L3) | 41 184 ns |
| mlp_fwd large (DRAM) | 1 178 790 ns |

## Expected Linux results (derived from step 10 roofline analysis)

The roofline model predicts:

**Bandwidth-bound kernels** (AI < 3.73 FLOP/byte — below the ridge):

| Kernel | Expected IPC | Expected L3 miss % | Explanation |
|---|---|---|---|
| dot_f32 (L1, 8KB) | > 3.0 | < 1% | 16 KB working set, deep in L1; FPU is the limit |
| dot_f32 (DRAM, 64MB) | < 1.5 | > 60% | Load-use stalls dominate; LSU saturated by DRAM traffic |
| matvec (L2, 128×128) | 2–3 | < 5% | Matrix fits in L2; some L1 capacity pressure from row widths |
| matvec (DRAM, 1024×1024) | < 1.5 | > 50% | 4 MB matrix → most rows evict from L3 before reuse |
| relu/add (DRAM) | < 1.5 | > 60% | Pure streaming; limited by DRAM read+write bandwidth |

**Compute-bound kernels** (AI > 3.73 — above the ridge):

| Kernel | Expected IPC | Expected L3 miss % | Explanation |
|---|---|---|---|
| matmul_naive 256×256 | 3–4 | < 5% | 768 KB working set in L3; FPU throughput is the limit |
| matmul_tiled T=64 256×256 | 3.5–4.5 | < 3% | Better L1/L2 reuse → slightly higher IPC and lower L1 miss % |
| matmul_tiled T=32 256×256 | 3.5–4.5 | < 2% | T=32 tile (12 KB) fits fully in L1; near-zero L1 miss % |

**Anomaly to watch for — T=48:**

From step 8 we know T=48 is 4–5× slower than T=32 or T=64. On Linux the counter data should show an L1 miss rate spike at T=48 vs its neighbours, caused by cache-set conflict misses from the 192-byte (non-power-of-2) tile stride interfering with the 32KB 8-way L1.

## How to run on Linux

```bash
# On AWS c5.2xlarge or c6i.2xlarge (x86_64, Linux):
sudo sysctl -w kernel.perf_event_paranoid=1

cmake --preset release
cmake --build --preset release --target perf_counters_bench
./build/release/cpu_engine/bench/perf_counters_bench
```

For independent validation with `perf stat`:
```bash
# dot_f32 cold (DRAM):
perf stat -e instructions,cycles,L1-dcache-load-misses,LLC-load-misses \
    ./build/release/cpu_engine/bench/perf_counters_bench 2>&1 | head -20
```

## Interpretation guide

| Pattern | IPC | L3 miss % | Diagnosis |
|---|---|---|---|
| Memory-bound (DRAM) | < 1.5 | > 50% | Load-use stalls; consider tiling or prefetch |
| Memory-bound (L3) | 1.5–2.5 | 10–50% | Spilling L2; check working set size |
| Compute-bound (not at peak) | 2.5–3.5 | < 5% | Instruction throughput; check SIMD width |
| At or near peak | > 4.0 | < 1% | Excellent; FMA units fully utilised |
| Branch-heavy | variable | low | High branch miss % → branchless patterns |
