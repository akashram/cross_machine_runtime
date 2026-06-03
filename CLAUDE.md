# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Phase 1 complete (steps 1–18)
All 18 steps done. Next: scaffold phases 2–12 with real sample implementations.

## Current status
- Phase 1, step 1 (CMake skeleton) — done
- Phase 1, step 2 (TSC benchmarking harness, `foundation/bench/bench.h`) — done
- Phase 1, step 3 (SPSC ring buffer, `foundation/spsc_queue.h`) — done. TSan clean. Benchmarks: 10 ns roundtrip (1T), 106 M items/sec throughput, 44 ns ping-pong RTT (release).
- Phase 1, step 4 (MPMC ring buffer, `foundation/mpmc_queue.h`) — done. TSan clean. Benchmarks: 21 ns roundtrip (1T), 52 M items/sec 1P-1C, 29 M 2P-2C, 12 M 4P-4C (release). 2x overhead vs SPSC in 1P-1C due to CAS.
- Phase 1, step 5 (ABA problem, `foundation/aba/`) — done. BuggyStack demo + AbaStack with 16-byte tagged-pointer CAS (cmpxchg16b). 25 ns (8B CAS) vs 39 ns (16B CAS). TSan clean.
- Phase 1, step 6 (hazard pointers, `foundation/hazard/`) — done. HazardDomain (global retire list, seq_cst slots, domain ID for tl() safety) + HazardStack<T>. 214 ns roundtrip. TSan clean.
- Phase 1, step 7 (epoch-based reclamation, `foundation/epoch/`) — done. EpochDomain (3-slot epoch cycling, global retire list) + EpochStack<T> (plain CAS — EBR prevents ABA implicitly). 171 ns roundtrip. TSan clean. Design doc in header: when to choose EBR vs hazard pointers.
- Phase 1, step 8 (RCU, `foundation/rcu/`) — done. RcuDomain (per-thread even/odd counter, synchronize() waits for active readers) + RcuPtr<T> (read-mostly atomic pointer). Read side: 31 ns (2× seq_cst fetch_add + acquire load). Write amortized: 58 ns (exchange + retire batch, synchronize() every 64 retires). TSan clean. Design doc in header: when to choose RCU vs EBR.
- Phase 1, step 9 (lock-free freelist, `foundation/freelist/`) — done. FreeList<T> with index-based next_ array (avoids union aliasing TSan race), 64-bit packed (idx, tag) head CAS, 32-bit ABA tag. acquire()+release(): 24 ns (2× LOCK CMPXCHG). macOS malloc is faster for tiny objects (9.5 ns TLS magazine); FreeList advantage is bounded latency + contended multi-thread. Design doc: why tagged pointers not HP for pool recycle. TSan clean.
- Phase 1, step 10 (Michael-Scott queue, `foundation/msqueue/`) — done. MsQueue<T> with embedded HazardDomain (2 HP slots: head + head->next). sentinel/data split via optional<T>. drain_one() helper avoids requiring default-constructible T in destructor. Single-thread: 250 ns enq+deq (malloc-dominated). 1P-1C: 211 ns/item (5 M/sec). TSan clean. Design doc: no tagged pointers needed (HP prevents ABA), copy-before-CAS avoids concurrent write race on node data.
- Phase 1, step 11 (Chase-Lev deque, `foundation/chase_lev/`) — done. ChaseLevDeque<T> (T must be trivially copyable). Owner push/pop from bottom (LIFO, no CAS in common case). Any thread steals from top (FIFO, one seq_cst CAS). Dual seq_cst fences in pop()+steal() for ARM correctness. Array growth via trash list (freed at destruction). push+pop: 34 ns (fence-dominated). steal 1T: 34 ns, 4T: 90 ns. TSan clean. 6 tests including last-element race (10K rounds).
- Phase 1, step 12 (work-stealing thread pool, `foundation/ws_pool/`) — done. WorkStealingPool (per-worker Chase-Lev deques, shared inbox for external submits, release-store bottom in push() for TSan visibility). TaskGroup (atomic remaining counter + condvar). parallel_for. 3 bugs fixed during TSan: (1) workers racing on workers_.size() during construction → two-phase init; (2) lost-wakeup in execute() notify without done_mu_ → hold mutex; (3) TSan false positive from fence+relaxed-bottom in Chase-Lev → changed to release-store bottom. ~470 ns/task overhead. 2.2x speedup on 4-worker vector sum. TSan clean. 7 tests.
- Phase 1, step 13 (coroutine execution engine, `foundation/coro/`) — done. Task<T> (lazy, symmetric transfer, variant result), AwaitableEvent (3-state atomic, set/suspend race), AwaitableMutex (cppcoro-style, intrusive Waiter on coroutine frame, release-store on Waiter::next to fix TSan ABA false positive). 12 tests. TSan clean. Three real bugs caught: (1) done.wait() before pool.wait() left frames live during ~Task(); (2) co_await mu.lock() as bare statement unlocks immediately (Guard is a temporary); (3) TSan ABA false positive on Waiter::next resolved by making it atomic<Waiter*>.
- Phase 1, step 14 (arena allocator, `foundation/arena/`) — done. Arena (mmap bump, MAP_HUGETLB on Linux / graceful fallback on macOS, MADV_DONTNEED/MADV_FREE_REUSABLE on reset), SizeClassedArena (8 power-of-2 classes 8–1024B, intrusive freelist per class, bit_ceil+ctz lookup), ThreadLocalArena (thread_local backing, zero-atomic fast path). 13 tests, TSan clean. Benchmarks (release): bump alloc 9.7 ns, freelist alloc+free 10.1 ns, malloc+free 59.3 ns (6x faster). Mixed sizes: arena 20.7 ns vs malloc 80.5 ns.
- Phase 1, step 15 (NUMA-aware allocator, `foundation/numa/`) — done. NumaTopology (sysfs on Linux, sysctl on macOS, cpulist parser), bind_thread_to_node (pthread_setaffinity_np on Linux / advisory hint on macOS), NumaArena (per-node SizeClassedArena, mbind via SYS_mbind syscall on Linux — no libnuma dep). 10 tests pass, 2 skip on single-node (multi-node test and concurrent test both skip correctly — real race if run on macOS since all threads map to node 0). TSan clean. Bench: 11.3 ns on macOS (same as local arena, no cross-node). Cross-node bench stub runs on Linux 2-socket.
- Phase 1, step 16 (unified tensor handle v1, `foundation/tensor/`) — done. TensorHandle: shared_ptr ref-counted buffer, void* data (offset for views), byte strides (not element strides), Dtype enum (8 types), Device enum (CPU/CUDA/FPGA stubs). Zero virtual dispatch — dtype dispatch is caller-side via switch(dtype()). Views: transpose (swap strides), slice (offset data + rescale stride), reshape (requires contiguous). element access via at<T>(indices). 14 tests, zero warnings, TSan clean. Bug caught: std::aligned_alloc on macOS rejected size/alignment combos — replaced with malloc (16-byte system alignment covers all dtypes ≤ 8 bytes).
- Phase 1, step 17 (property-based testing, `foundation/proptest/`) — done. Minimal QuickCheck-style framework: splitmix64 RNG (deterministic by seed), Gen<T> generator type (complexity-scaled), shrink() specializations for int/size_t/vector<T>, greedy shrink loop, check() template. Properties verified: SPSC FIFO + no-loss, MPMC no-loss (3P×3C), freelist bounded + no-loss, MsQueue FIFO + no-loss, arena non-overlap + reset-rewinds, TensorHandle numel/strides/transpose-involution/reshape. TSan clean. Bug caught: MPMC consumers deadlocked because only one thread hit the break condition — fixed by using a shared atomic load as exit condition.
- Phase 1, step 18 (hardware counter infrastructure, `foundation/perf/`) — done. PerfCounters: perf_event_open() event group (cycles/insn/LLC-refs/LLC-misses/branches/branch-misses), PERF_FORMAT_GROUP atomic read, multiplexing correction via time_enabled/time_running ratio, exclude_kernel=1. macOS: RDTSC fallback, available()=false. measure_perf() helper. bench_main integrated: shows IPC/L3miss/Brmiss on Linux; macOS prints expected values for reference. 6 tests, TSan clean.

## Tooling decisions
- **Compiler:** Apple clang 14. No pre-built LLVM binaries exist for Intel macOS — brew always builds from source (2-5 hrs). Apple clang handles C++20/23 and all sanitizers fine for Phase 1.
- **clang-tidy:** Deferred. Will be set up when LLVM is built from source in Phase 4 step 1 (that build is intentional — it's a learning exercise).
- **Build system:** Ninja. CMake generates Ninja files.
- **Presets:** debug, release, asan, tsan, ubsan — each builds to its own `build/<preset>/` directory. Run with `cmake --preset <name>` and `cmake --build --preset <name>`.

## Execution plan (agreed 2026-05-31)
1. Finish Phase 1 fully (steps 5–18) with real implementations and tests.
2. Scaffold all phases 2–12 with real sample implementations — actual CUDA kernels, Raft, ZeRO, GBDT, etc. Hardware-gated with #ifdef/find_package where toolchain absent. Written as a safety checkpoint; user will walk through code with Claude to understand it.
3. Work through phases one by one to build real understanding.

## Phase 12: Machine Learning (added 2026-05-31)
Inserted after Phase 6 (distributed training), before Phase 9 (inference serving). See PLAN.md for full detail.
- **12a — Classical ML:** Random Forest, GBT (XGBoost/LightGBM-style), SVM, k-NN, k-means++, PCA
- **12b — Decision Framework + Benchmarks:** cross-method ablation on OpenML CC-18, written decision criteria, ensemble composition rules, hyperparameter sensitivity analysis
- **12c — Hyperparameter optimization:** Bayesian opt (GP), TPE, Hyperband/ASHA
- Benchmark targets: beat LightGBM throughput on large tabular while matching accuracy; compete on MLPerf
- Design docs are first-class: each algorithm family gets decision criteria, math foundations, failure modes

## Non-negotiable standards (from PLAN.md)
- Every component is benchmarked before moving on. No "I'll benchmark later."
- Every data structure has property-based tests verifying linearizability.
- TSan must find zero races on all lock-free data structures.
- Every non-obvious decision gets a written design doc before implementation.
- Hardware counter data (IPC, cache miss rates) on every CPU benchmark.
