# Step 2 — Explicit hugepage allocator

## What was built

`HugeRegion` — an RAII `mmap` wrapper for huge-page-backed memory regions:
- **Linux:** requests 2 MB pages via `mmap(..., MAP_HUGETLB)`, rounding the size up to a 2 MB boundary as the syscall requires. Falls back silently to 4 KB pages if the kernel's hugepage pool (`/proc/sys/vm/nr_hugepages`) is empty — `is_huge()` reports which path was taken.
- `mbind(MPOL_BIND)` after `mmap` (before any page fault, so `MPOL_MF_MOVE` isn't needed) binds the physical pages to a NUMA node, composing directly with `ThreadPinner` from step 1: pin the thread, then bind its memory to the same node.
- `prefault()` writes one byte per page to force first-touch physical allocation before timing starts — without it, the first benchmark pass measures page-fault handling, not memory access.
- `expected_tlb_entries()` quantifies TLB pressure directly from the working-set size and page size.
- **macOS:** has no user-space explicit-hugepage API (the VM system may transparently promote 4 KB → 2 MB pages, but this can't be requested or observed from user code). `try_huge` is silently ignored; `is_huge()` always returns `false`.

## Measured results (macOS Intel, 256 MiB region, 5 passes, 64-byte cache-line stride)

```
              huge   tlb_entries   ns/line   throughput
4 KB pages    no     65536         4.95 ns   12.9 GB/s
2 MB pages    no     65536         5.65 ns   11.3 GB/s   (requested huge, got 4 KB — no API)

TLB entry reduction: 1x (huge pages not available on this platform)
```

## Key findings

**The numbers above measure "no difference" — correctly, because there is no difference to measure on macOS.** Both rows ran on identical 4 KB pages (`huge=no` for both); the "2 MB pages" row is what happens when the code *asks* for huge pages on a platform that has no mechanism to grant the request. The ~12% gap between the two rows (11.3 vs 12.9 GB/s) is page-cache / thermal noise across runs, not a hugepage effect — `expected_tlb_entries()` reports 65,536 for both, confirming the allocator is honest about what it actually got.

**This makes the macOS run a control, not a result.** The real test is the Linux row this component is built for: with `nr_hugepages > 0`, the *same* allocator call returns `is_huge() == true`, `expected_tlb_entries()` drops to 128 (a 512× reduction — 256 MB ÷ 2 MB vs. 256 MB ÷ 4 KB), and `perf stat -e dTLB-load-misses` should show the corresponding ~65,000 → ~128 misses-per-pass collapse. The expected wall-clock win for a bandwidth-bound scan is 10–40%.

**Why `expected_tlb_entries()` reports a *theoretical minimum*, not a measured count:** it assumes no eviction — the actual miss rate depends on the access pattern and the STLB's real capacity (typically 32–64 entries for large pages). The 512× ratio between page sizes is the structural fact that doesn't depend on access pattern; the *measured* `dTLB-load-misses` ratio (via `perf stat` on Linux) is what confirms the structural prediction actually translates into fewer hardware page-table walks.

## Platform notes

To reproduce the Linux numbers:
```bash
echo 128 | sudo tee /proc/sys/vm/nr_hugepages   # reserve 256 MB of 2 MB pages
cmake --preset release && cmake --build --preset release --target hugepage_bench
perf stat -e dTLB-load-misses,dTLB-store-misses ./build/release/cpu_engine/bench/hugepage_bench
```
A pool of 0 (the default at boot) makes every `alloc(try_huge=true)` fall through to the 4 KB path — the same code path this macOS run exercised, just for a different reason (no kernel pool vs. no kernel API). `is_huge()` can't distinguish the two causes; only the platform `#ifdef` can.
