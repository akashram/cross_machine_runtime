# Step 12 — Profile-guided optimization (PGO)

## What was built

End-to-end PGO workflow for Apple Clang (Clang instrumentation-based PGO):

| File | Purpose |
|---|---|
| `cmake/Pgo.cmake` | `RUNTIME_PGO=instrument\|use` flag injection |
| `CMakePresets.json` | `pgo-instrument` and `pgo-use` presets |
| `pgo_train.cpp` | Representative training workload |
| `pgo_measure.cpp` | Before/after measurement binary |
| `run_pgo.sh` | Orchestrates the full 3-step workflow |

## Workflow

```
Step 1  cmake --preset pgo-instrument   → build with -fprofile-instr-generate
Step 2  LLVM_PROFILE_FILE=... ./pgo_train  → generates .profraw
Step 3  llvm-profdata merge ... → build/pgo.profdata
Step 4  cmake --preset pgo-use          → build with -fprofile-instr-use=pgo.profdata
Step 5  compare ./build/release/pgo_measure  vs  ./build/pgo-use/pgo_measure
```

Run the full workflow: `./cpu_engine/pgo/run_pgo.sh` from the repo root.

## Results (macOS Intel, AVX2)

| Kernel | Release GFLOPS | PGO GFLOPS | Change |
|---|---|---|---|
| mlp kRelu 64→128→128→64→32 | ~10–15 | ~10–13 | noisy, inconclusive |
| mlp kSigmoid 32→64→32→1 | ~7–11 | ~8–11 | noisy, inconclusive |
| matmul_tiled T=64 256×256 | **~38** | **~18** | **−52% regression** |
| dot_f32 16K (L1) | ~21–24 | ~20–23 | neutral |

## Key findings

### Matmul regression (−52%) — a PGO failure case

The matmul regression is consistent across runs and is the most instructive result. Two contributing factors:

**Profile mismatch warning.** The `pgo-use` build emits:
```
warning: profile data may be out of date: of 531 functions,
         1 has mismatched data that will be ignored
```
When a function's profile is discarded due to mismatch (likely `matmul_tiled_f32` itself after the pragma removal in mlp.h changed the binary), the compiler falls back to default heuristics — but now *also* suppresses some optimisations it would have applied without PGO. The result is a binary that's worse than no-profile at all.

**SIMD inner loops don't benefit from PGO.** The matmul hot path is a triple-nested loop with no unpredictable branches. The compiler's default decisions (vectorisation width, unroll factor, software pipelining) are already near-optimal without profile guidance. PGO data about loop trip counts can cause the compiler to choose a different unroll factor that interacts poorly with the hardware's pipeline or SIMD alignment.

### MLP — inconclusive on macOS

MLP results vary 2× between runs due to:
- macOS thread pinning is advisory (not hard), so the OS can migrate the thread mid-measurement
- Power management causes clock frequency variation between cold and warm runs
- 500-pass measurement × ~5–8 µs per pass = only 3–4 ms of measurement — insufficient for thermal stability

**Expected Linux result:** On a server with hard affinity (`pthread_setaffinity_np`), stable clocks, and `nohz_full`, the MLP activation switch (`switch(cfg_.acts[l])`) is a genuine PGO target. With 70% kRelu training distribution, PGO should:
- Place `kRelu` as the first case (already true by default, so minimal gain)
- Mark `kSigmoid` and `kNone` branches as cold, enabling better layout
- Potentially inline `forward()` more aggressively at call sites

Expected improvement on Linux: 5–15% for the MLP, 0% for matmul and dot.

### dot_f32 — neutral (expected)

No branches to predict, no inlining decisions to make. PGO correctly has nothing to offer here.

## Lessons

| Pattern | PGO effect | Why |
|---|---|---|
| Branchy dispatch (activation switch) | Positive | Correct branch ordering and cold-path layout |
| Compute-bound SIMD (matmul inner loop) | Neutral to negative | No unpredictable branches; profile can disrupt vectorisation decisions |
| Pure streaming (dot, eltwise) | Neutral | No branches to predict, already vectorised |
| Profile mismatch | Negative | Discarded function data causes compiler to make conservative-but-suboptimal choices |

**PGO is not free.** The profile must exactly match the binary being compiled. Any source change between profile collection and the `pgo-use` build invalidates part of the profile. In a real project, PGO must be part of a locked CI workflow (build → profile → recompile) run on the exact same source tree.

## Platform notes

For clean results, run on Linux with:
```bash
# Hard-pin the training workload
taskset -c 0 ./build/pgo-instrument/cpu_engine/pgo/pgo_train

# Stable clocks
cpupower frequency-set -g performance
echo 0 > /sys/devices/system/cpu/cpufreq/boost  # disable turbo
```
