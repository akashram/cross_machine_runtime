# Runtime Dialect (Design + Registration)

**Status: code-complete, not yet built — requires MLIR built from source on Linux.**

## What this measures
Defines and registers the `runtime` MLIR dialect: 15 ops (`matmul`, `conv`,
elementwise unary/binary, `softmax`, `reduce`, `gather`/`scatter`, plus the
pass-inserted `fusion_group`/`yield`, `transfer`, `all_gather`/
`reduce_scatter`, `kernel_call`), three attributes (`DeviceAttr`,
`ReduceKindAttr`, `ShardSpecAttr`), and an `InferTypeOpInterface`
implementation for every shape-derivable op. Validates round-trip:
parse → print → parse produces identical IR.

## Design decisions
- **Device/shard as discardable attributes, not part of the type.** Ops are
  authored device-agnostic; `runtime.device` and `runtime.shard` are set by
  the placement (step 9) and sharding (step 10) passes after the fact. This
  keeps shape inference, fusion, memory planning, and remat free to reason
  about pure dataflow without a device axis, and lets placement re-run
  without changing every op's type signature. See `compiler/DESIGN.md` §2.
- **`fusion_group` is a single-region op**, not one new op per fusable
  pattern (à la `linalg.generic`). The fusion pass (step 5) matches a
  subgraph and moves it into the region wholesale; `fusion_kind` is a string
  key kernel specialization (step 11) looks up to find the fused kernel
  implementation.
- **`InferTypeOpInterface` lives on the op, not the pass.** `matmul` and
  `conv`'s `inferReturnTypes` are hand-written (contraction-dim / spatial
  arithmetic); elementwise ops get it via `SameOperandsAndResultType` or a
  small broadcast helper. The shape-inference pass (step 4) is then just a
  topological walk that calls the interface — no op-specific logic in the
  pass itself.

## Results

TODO: run on Linux with MLIR and fill in this table.

| Test | Result |
|------|--------|
| TableGen compiles cleanly | TODO |
| `mlir-opt --load-dialect-plugin` loads runtime dialect | TODO |
| Round-trip test (parse → print → parse) | TODO |
| Assembly format: `runtime.matmul` prints correctly | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built (see mlir_setup/)
- Build preset: compiler CMake with MLIR_DIR set
