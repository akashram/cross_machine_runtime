# Affine Dialect Lowering

**Status: STUB — requires MLIR on Linux.**

## What this measures
Lowers runtime dialect loop nests to MLIR Affine dialect for polyhedral
analysis. Applies affine tiling and loop interchange transformations.

## Results
TODO: run on Linux with MLIR.

| Transformation | Loop nest | Tile size | Cache miss reduction |
|----------------|-----------|-----------|---------------------|
| Tiling (matmul) | 3-level | TODO | TODO |
| Interchange | matmul ijk→ikj | N/A | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
