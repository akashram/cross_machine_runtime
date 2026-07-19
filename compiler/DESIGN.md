# Compiler / IR (MLIR) Architecture

Status: Phase 4 code-complete (15/15 steps), zero steps run — no MLIR/LLVM
toolchain on the Mac this was written on. Every pass compiles only against
a Linux build with `MLIR_DIR` set (see `mlir_setup/build_llvm.sh`), except
`cost_model/`, which has no MLIR dependency and is the one part of this
phase that's actually been built and run (see `cost_model/README.md`).
This document covers the decisions made *before* a build/run pass is
possible; per-step READMEs hold the (still-TODO) measured numbers.

---

## 1. Why device/shard live as discardable attributes, not in the type

The alternative considered: a custom `RuntimeTensorType` carrying device and
sharding as type parameters (`!runtime.tensor<device=gpu, shard=<...>, 4096x4096xf32>`),
the way some MLIR-based ML compilers do it.

**Why not:** every pass before placement (shape inference, fusion, memory
planning, remat) has nothing to say about device or sharding — they reason
about pure dataflow. Baking device/shard into the type means those passes
either ignore a type parameter they never touch (attribute pollution) or,
worse, have to preserve it correctly through every type-producing
transformation (fusion's `inferReturnTypes`, remat's cloned ops) even
though it's not their concern. A type change also means every op's result
type signature shifts the moment placement runs, which is exactly the kind
of "passes couple to each other's output shape" problem this project's
pass pipeline is trying to avoid (see Phase 4's build order — nine
independent, individually testable passes).

Discardable attributes (`runtime.device`, `runtime.shard`) sidestep this:
ops are authored and shape-inferred fully device/shard-agnostic; placement
(step 9) and sharding (step 10) attach attributes as a strictly additive
final pass over already-correct IR. `getAssignedDevice`/`setAssignedDevice`
in `RuntimeDialect.h` are the only sanctioned way to read/write them, so if
a future pass needs the type-carrying approach instead, the blast radius is
those two functions' call sites, not every op definition.

## 2. `fusion_group`: one region-based op, not one op per fused pattern

The fusion pass (step 5) needs to represent "matmul + bias + relu, as one
scheduling unit" without inventing `Runtime_FusedMatmulBiasReluOp`,
`Runtime_FusedMatmulBiasGeluOp`, `Runtime_FusedConvBiasReluOp`, ... for
every pattern the pass might ever match. `runtime.fusion_group` holds the
matched subgraph in a single-block region (the same shape as `scf.if`), and
`fusion_kind` is a string key kernel specialization (step 11) looks up. The
cost: kernel_spec must maintain a `fusion_kind -> symbol` table instead of
one entry per fused op type. That's the right tradeoff here — the set of
*matchable patterns* grows with the fusion pass's sophistication, but the
*op vocabulary* doesn't need to grow in lockstep.

## 3. Placement (step 9) is greedy, not a global optimum

Per-op cost minimization considers only that op's compute cost plus
transfer-in cost from already-placed producers — it does not look ahead to
how placing op N on GPU constrains op N+1's transfer cost. A real
DP/ILP-based scheduler (weighing the whole op DAG, as XLA's SPMD partitioner
or a Halide-style autoscheduler would) is real, known future work, not an
oversight — it's not worth building against a cost model whose constants
are still spec-sheet placeholders (see §5). Greedy placement is enough to
validate the mechanic that matters right now: attribute assignment +
explicit `runtime.transfer` insertion, both of which the global-optimum
version would still need.

## 4. Rematerialization (step 8) is single-op granularity

True activation checkpointing recomputes a whole forward *segment* during
backward. This pass instead clones one cheap op (relu/gelu/sigmoid/
add/mul/sub) at each of its uses past the first, shrinking that one value's
live range. It's the mechanical primitive a training-loop-aware pass in
Phase 6 would call repeatedly once there's a real forward/backward boundary
to reason about — building the segment-level version now, against no real
training loop IR, would mean guessing at an interface Phase 6 will define.

## 5. The cost model's constants are spec sheets, not measurements — and
   that's why it's the one part of Phase 4 not gated behind `MLIR_DIR`

`cost_model/CostModel.cpp` has zero MLIR types in it — `estimate_us()` is
`max(flops/peak_flops, bytes/peak_bandwidth) + launch_overhead`, the same
roofline combinator used throughout Phases 2 and 3. Because it's pure C++,
it's wired into the root `CMakeLists.txt` unconditionally instead of behind
the `MLIR_DIR` gate the rest of `compiler/` sits behind, and it's the one
Phase 4 component that's actually been compiled and run (on the Mac, via
plain `clang++`) — see `cost_model/README.md` for the captured output. The
placement and sharding passes are structurally ready for calibrated numbers
the moment Phases 2/3/7/8 produce them; nothing about the pass logic
changes, only `get_device_cost()`'s return values do.

## 6. Kernel specialization emits real symbol references, not strings

`runtime.kernel_call`'s `callee` is a `FlatSymbolRefAttr`, and kernel_spec
(step 11) inserts a matching `func.func private` declaration into the
module for every symbol it references — `cpu_engine::avx512::matvec_f32`,
`gpu_engine::kernels::gemm_wmma`, etc. become real, `SymbolTable`-verifiable
declarations, not opaque strings the AOT pipeline has to trust blindly.
This is what makes `runtime_aotc`'s final link step (step 12) a normal
"link object files against extern declarations" problem instead of a
custom resolution mechanism.

## 7. Ops with no kernel_spec table entry are left unlowered, not an error

A module can legitimately reach kernel specialization with ops this project
hasn't written a kernel for yet (or a TPU placement, which lowers through
StableHLO in Phase 8, not a direct `kernel_call`). Failing the pass on the
first unmatched op would make every partial build all-or-nothing;
`runtime.kernels_lowered`/`runtime.kernels_skipped` module attributes make
the partial state visible and queryable (the AOT CLI prints both) instead
of the pass silently doing nothing or hard-failing.
