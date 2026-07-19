# Affine Dialect Lowering

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Lowers runtime dialect loop nests to MLIR Affine dialect for polyhedral
analysis. Applies affine tiling and loop interchange transformations.

## Design
Two passes (`AffinePass.cpp`): `runtime-affine-lower` converts a rank-2,
fully-static, bias-free `runtime.matmul` into an `affine.for` triple-nest
over `bufferization.to_memref`-bridged buffers (naive ijk order — this pass
only establishes correctness); `runtime-affine-tile` then runs
`affine::getTileableBands` + `affine::tilePerfectlyNested` over whatever
bands exist, uniform tile size on every loop dim. Batched (rank > 2) and
dynamic-shape matmuls are left as `runtime.matmul` — affine loop bounds must
be compile-time constants, so kernel specialization (step 11) dispatches
those as an opaque call instead. Loop interchange (ijk → ikj) is scoped as
a follow-up on the tiled point loops once real cache-miss numbers justify
which of j/k should be innermost — see the comment in `AffineTilingPass`.

## Results
TODO: run on Linux with MLIR.

| Transformation | Loop nest | Tile size | Cache miss reduction |
|----------------|-----------|-----------|---------------------|
| Tiling (matmul) | 3-level | TODO | TODO |
| Interchange | matmul ijk→ikj | N/A | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
