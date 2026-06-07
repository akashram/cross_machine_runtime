# Step 4 — Non-temporal store primitives

## What was built

`nt_store.h` provides:
- `sfence()` — the mandatory store fence after an NT sequence (NT stores bypass the normal cache-coherent ordering; nothing guarantees visibility to other cores until `sfence` retires).
- `nt_store<T>()` — raw typed non-temporal stores (`int32`, `int64`, `__m128i`, `__m256i`, `__m256`, `__m256d`) with **no implicit fence** — the caller controls batching and decides when ordering actually matters.
- `nt_memset(dst, byte, size)` — AVX2 path (4×32B/iteration via `_mm256_stream_si256`) when available, SSE2 fallback (4×16B/iteration), scalar for non-x86; ends with `sfence()`. Unaligned heads are handled with scalar stores before entering the 32-/16-byte-aligned main loop.
- `nt_memcpy(dst, src, size)` — streaming copy: source is read normally (so the hardware prefetcher can do its job), destination is written with NT stores (to avoid polluting the cache with data that will never be read back) — the canonical "DMA staging buffer" pattern.

## Measured results (macOS Intel, AVX2, 512 MiB buffer, 3 passes)

```
Scenario A — cold write (buffer >> L3, both paths must hit DRAM):
  regular memset (cold)    50.3 GB/s
  NT memset (cold)         17.5 GB/s        →  NT is 2.9x SLOWER

Scenario B — warm-ish write (src read first; 512 MB still >> L3 either way):
  regular memset (warm)    47.7 GB/s
  NT memset (warm)         18.3 GB/s        →  NT is 2.6x SLOWER

Scenario C — memcpy 512 MiB src→dst (staging-buffer pattern):
  regular memcpy            6.5 GB/s
  NT memcpy                 6.9 GB/s        →  NT is ~6% faster (the one win)
```

## Key findings — the most important negative result in this component

**NT stores are not "the fast way to write memory" — they are *2.9× slower*
than the standard library for zero-fill on this chip, and the reason is
mechanical, not incidental.** `memset` on Broadwell+ Intel compiles down to
`rep stosb`, which the microcode implements via **ERMSB** (Enhanced REP
MOVSB/STOSB) — a hardware-accelerated streaming-store path that is itself
non-allocating (it already skips the read-for-ownership / RFO that NT
stores exist to avoid) but can drive the full width of the store datapath
in a way that a manually-unrolled `_mm256_stream_si256` loop, bound by
per-instruction issue throughput, cannot match. **The textbook advice
("use `_mm_stream_*` for write-only paths to skip RFO") assumes the
baseline you're beating *does* RFO — on this hardware, the baseline you'd
be "improving on" already doesn't.**

This is why the benchmark's decision guide inverts the naive rule:

| Use NT stores when | Prefer `memset`/`memcpy` when |
|---|---|
| Non-zero fill (ERMSB is zero-fill-specific) | Pure zero-fill on modern Intel |
| No ERMSB (older x86, some ARM) | Data will be read back soon (NT bypass *causes* a cache miss on the read side — you'd be paying for the pollution-avoidance you didn't need) |
| Building a DMA staging buffer another thread/device consumes | — |
| `sfence`-based ordering is the actual requirement (lock-free producer signalling), not raw throughput | — |

**Scenario C is the one case where NT wins, and it wins for the textbook
reason** (~6%, 6.9 vs 6.5 GB/s): a copy whose destination will not be
touched again benefits from not evicting the rest of the working set —
there's no competing ERMSB-equivalent fast path for `memcpy`'s
write side once the source must also be streamed in.

**Lesson generalized in the design doc:** always benchmark a hand-rolled
"faster" primitive against the compiler/libc baseline before adopting it —
a technique that's a documented win in a textbook can be a measured
regression on a specific microarchitecture, and the only way to know which
applies here is to run both and look.

## Platform notes

`-mavx2` is enabled for `cpu_engine` targets on x86_64; the SSE2 path
exercises on non-AVX2 x86, scalar on non-x86. On Linux, confirm the
cache-pollution claim directly:
```bash
perf stat -e cache-misses,cache-references,LLC-load-misses,dTLB-load-misses,iTLB-load-misses \
    ./build/release/cpu_engine/bench/nt_store_bench
```
Expect regular `memset`/`memcpy` to show higher `LLC-load-misses` on a
*subsequent* read of the same buffer (cache pollution), and NT variants to
show near-zero L1/L2 residency for the destination — the inversion of the
raw-throughput numbers above, and the actual justification for NT stores
in a producer/consumer pipeline where the consumer reads the data next.
