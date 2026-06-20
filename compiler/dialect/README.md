# Runtime Dialect (Design + Registration)

**Status: STUB — requires MLIR built from source on Linux.**

## What this measures
Defines and registers the `runtime` MLIR dialect with ops for matmul, conv,
elementwise, reduce, scatter/gather. Types carry device placement annotations.
Validates round-trip: parse → print → parse produces identical IR.

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
