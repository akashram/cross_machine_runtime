# Operator Fusion Pass

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Pattern-matches fusable op sequences ({matmul,conv} [+bias] + [relu|gelu|sigmoid])
and replaces them with a single `runtime.fusion_group`. Measures HBM traffic reduction.

## Design
`OpRewritePattern<ActOp>` per activation (Relu/Gelu/Sigmoid), templated over
the activation type so the same matching logic (`matchChain` in
`FusionPass.cpp`) serves all three. Matches walk backward from the
activation through an optional bias-`add` to a `matmul` or `conv`, requiring
every intermediate op have exactly one use — fusing a value with multiple
consumers would duplicate compute, so the pattern simply declines. On match,
the chain is cloned (via `IRMapping`) into a fresh `fusion_group` region
whose block arguments are the chain's external inputs (required since
`fusion_group` is `IsolatedFromAbove`), and the original ops are erased.
`fusion_kind` (e.g. `"matmul_bias_relu"`) is looked up by kernel
specialization (step 11) to pick the fused kernel implementation.

## Results
TODO: run on Linux with MLIR.

| Pattern fused | HBM reads (before) | HBM reads (after) | Reduction |
|---------------|--------------------|--------------------|-----------|
| matmul+bias+relu | TODO | TODO | TODO |
| matmul+bias+gelu | TODO | TODO | TODO |
| conv+bias+relu | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
