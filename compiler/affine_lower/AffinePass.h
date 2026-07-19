//===- AffinePass.h - Step 6: lower to Affine dialect + tile ------------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Lowers `runtime.matmul` (the only op with a genuinely interesting loop
// nest at this stage — elementwise ops don't benefit from polyhedral
// analysis) to an `affine.for` triple-nest over freshly-allocated memrefs,
// bridging the tensor/memref boundary with `bufferization.to_memref` /
// `to_tensor`. Only applies to ops with fully static shapes: affine loop
// bounds must be compile-time constants (or affine expressions of them),
// so a dynamic dim here just isn't lowered — it stays a `runtime.matmul`
// for kernel specialization (step 11) to dispatch as an opaque call.
std::unique_ptr<mlir::Pass> createAffineConversionPass();

// Applies affine loop tiling to every perfectly-nested affine.for band
// produced above, with a uniform tile size on every loop dimension.
// Interchange (documented in the design doc as ijk -> ikj for matmul) is
// a manual permutation applied to the tiled band, not a separate pass —
// tiling and interchange share the same "band" abstraction so doing them
// as two passes would mean re-discovering the band twice.
std::unique_ptr<mlir::Pass> createAffineTilingPass(unsigned tileSize);

} // namespace runtime
