# vliw_analysis

**Status: script done, toolchain/hardware-gated for the TPU-specific output
— the TPU number stays unrun. Now actually run on CPU (2026-08-01, JAX
0.4.38 in `.venv/`, see below) as a smoke test of the JAX toolchain itself,
not a substitute for the TPU measurement. Design analysis below is written
from documented TPU architecture (the TPU v4 paper describes the vector
unit's VLIW ISA), to be checked against real dumped text once available,
not against unverifiable numbers.**

## What this measures

PLAN.md Phase 8 step 10: inspect XLA-generated HLO for a kernel, understand
the instruction bundling, document how this differs from x86 OOO and
NVIDIA SIMT.

## Design

`dump_hlo.py` dumps two levels for a representative matmul+bias+relu
kernel: `lower(...).compiler_ir(dialect="hlo").as_hlo_text()` (backend-
agnostic HLO — the same IR XLA would produce compiling for CPU) and
`lower(...).compile().as_text()` (the compiled executable's backend-
specific text, where VLIW-bundle structure would actually show up on the
TPU backend — this call requires compiling for `tpu` specifically; run on
CPU it shows LLVM IR instead, not the thing this step is asking about).

## x86 OOO vs. NVIDIA SIMT vs. TPU VLIW — how the same "keep functional
units busy" problem gets solved three different ways

| | x86 OOO | NVIDIA SIMT | TPU vector unit (VLIW) |
|---|---|---|---|
| Who finds ILP | hardware, at runtime | hardware (warp scheduler), at runtime, across threads not instructions | compiler (XLA), at compile time |
| Mechanism | reorder buffer + register renaming reorders a sequential stream | many resident warps hide any single warp's stall by switching to another | a single wide instruction bundle statically assigns one op per functional-unit slot per cycle |
| What happens when scheduling is wrong | hardware absorbs it — stalls a cycle, keeps going | a warp stalls, scheduler just runs a different resident warp instead | **the bundle has an idle slot for that cycle, and there is no hardware fallback to fill it** |
| Divergent control flow | branch prediction + speculation | predication + reconvergence across lanes in a warp | XLA fuses/schedules a straight-line dataflow graph ahead of time; TPU has comparatively little support for data-dependent control flow inside a kernel — this is *why* runtime.fusion_group exists as a compile-time grouping mechanism (compiler/dialect) rather than something resolved dynamically |

The load-bearing difference for this repo: on x86 and (to a lesser extent,
via warp-level parallelism) on NVIDIA GPUs, a suboptimal instruction
schedule is a *performance* bug — the hardware still gets correct results,
just slower. On the TPU's VLIW vector unit, since there's no dynamic
scheduling hardware to compensate, a bad bundle packing is directly wasted
cycles with nothing hiding it. This is the same "the compiler owns
everything, there's no hardware safety net" theme `hbm_sram/README.md`
documents for HBM<->VMEM data movement — VLIW bundling is the instruction-
issue-level version of the same architectural philosophy: TPUs push
scheduling decisions other architectures make in hardware, at runtime,
into the compiler, at compile time, in exchange for silicon spent on more
MXU/vector throughput instead of scheduling logic.

## Local CPU smoke test (2026-08-01)

Ran `dump_hlo.py` unmodified against JAX 0.4.38 (CPU backend, `.venv/`).
Both dumps produced real output:
- The backend-agnostic HLO (`compiler_ir(dialect="hlo").as_hlo_text()`) is
  the clean, unfused `dot -> broadcast -> add -> maximum` graph expected.
- The "compiled executable text" (`compile().as_text()`) turned out to be a
  **scheduled, fused HLO module** (bf16<->f32 converts folded together, a
  `parallel_maximum_convert_fusion` kernel, `outer_dimension_partitions`
  backend_config annotations for CPU multi-threading) — this docstring's
  prediction that CPU output "would show LLVM IR structure instead" is
  slightly off; XLA's CPU backend's `as_text()` still emits HLO-level text,
  just post-fusion/post-scheduling, not literal LLVM IR. Doesn't change the
  step's conclusion (no VLIW bundle structure is visible either way on
  CPU), just a minor correction to the design note above, not yet fixed in
  the docstring itself.

This confirms the JAX trace/lower/compile pipeline itself works end to
end; it is not the TPU-backend measurement this step still needs.

## Results
TODO: run `dump_hlo.py` on a GCP TPU VM and confirm the compiled text
actually exposes VLIW bundle structure — Google does not publicly document
TPU's VLIW mnemonics, so it's possible `as_text()`'s TPU output stays at a
level of abstraction above visible bundling (worth confirming empirically
rather than assumed from the design table above).

| Kernel | HLO op count | Compiled text available? | VLIW bundling visible? |
|---|---|---|---|
| matmul(512,512) + bias + relu | TODO | TODO | TODO |

## Hardware notes
- Required: GCP TPU VM, for the TPU-backend `compile().as_text()` call
  specifically (the HLO dump works on any backend, including CPU).
