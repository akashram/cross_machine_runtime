# stablehlo_lower

**Status: code-complete, toolchain-gated — real MLIR `DialectConversion` pass,
unrun. Needs an MLIR build (compiler/mlir_setup, Linux-only per Phase 4)
*plus* StableHLO (github.com/openxla/stablehlo) built against the same LLVM
commit — neither exists on this Mac, so this is one gate deeper than the
rest of `compiler/`.**

## What this measures

PLAN.md Phase 8 step 3: a lowering pass from `compiler/dialect`'s `runtime`
dialect to StableHLO, the dialect JAX/XLA consume for TPU (and GPU/CPU XLA)
execution. Validated with `stablehlo-opt` per PLAN.md's step-3 gate.

## Design

`StableHLOLowerPass.cpp` is a `ConversionPattern`-based pass
(`applyPartialConversion`, `runtime` dialect marked fully illegal on exit)
covering every op in `RuntimeOps.td` except three deliberately left
unconverted:

| runtime op | StableHLO op(s) | Notes |
|---|---|---|
| `matmul` | `dot_general` | batch/contracting dims derived from rank + `transpose_lhs/rhs`, no materialized transpose |
| `add`/`mul`/`sub` | `add`/`multiply`/`subtract` | direct rename |
| `relu` | `maximum(x, 0)` | StableHLO has no relu primitive; this is the standard idiom |
| `sigmoid` | `logistic` | direct rename |
| `gelu` | `tanh`-approximation, decomposed | matches what most JAX/Flax models use; documented as a numerical choice, not the exact erf-based GELU |
| `softmax` | `exponential` / `reduce` / `divide`, decomposed | numerically-stable (subtract max first) |
| `reduce` (sum/max/mean) | `reduce` (+ `divide` for mean) | StableHLO has no reduce-mean primitive |
| `conv` | `convolution` | NCHW dimension numbers, matching `runtime.conv`'s documented layout |
| `all_gather` | `all_gather` | feeds step 8 (`ici_collectives`) |
| `reduce_scatter` | `reduce_scatter` | combiner hardcoded to sum — the only reduction kind PLAN.md's gradient/activation all-reduce steps need |

Left **not** converted, on purpose:
- `gather`/`scatter`: `stablehlo.gather`/`scatter` need full
  `GatherDimensionNumbers`/`ScatterDimensionNumbers` (and, for scatter, a
  combiner region) to express correctly. A shallow version would silently
  produce wrong indices for anything but the simplest case — real work,
  left as a TODO on top of the code-complete steps rather than shipped
  half-right.
- `transfer`/`kernel_call`: these are placement/kernel-specialization
  artifacts (steps 9/11) that shouldn't reach this pass on the TPU path —
  XLA does its own device placement inside a slice, and the runtime
  dialect's cross-device transfer/dispatch ops are for the CPU/GPU/FPGA
  backends. Left illegal deliberately, so an op reaching this pass on those
  paths surfaces as a conversion failure (wrong pass ordering upstream)
  instead of being silently mis-lowered.
- `fusion_group`: not handled here at all — per `StableHLOLowerPass.h`, a
  fusion group must be dissolved back to its constituent ops *before* this
  pass runs (XLA fuses during its own HLO compilation; re-fusing first
  would just have XLA fight or undo the earlier CPU/GPU-oriented decision).

## Results
TODO: validate with `stablehlo-opt --verify-diagnostics` on a sample
`runtime` dialect module (matmul+bias+relu, softmax, an all_gather) once
MLIR + StableHLO are built (Linux).

| Op sequence | Lowering result |
|---|---|
| matmul → bias-add → relu | TODO |
| softmax | TODO |
| all_gather | TODO |

## Hardware notes
- Required: Linux (LLVM/MLIR from source, per `compiler/mlir_setup`), plus
  StableHLO built against the identical LLVM commit — StableHLO does not
  version-pin to LLVM releases, only to specific commits, so this is a
  stricter requirement than the rest of Phase 4.
