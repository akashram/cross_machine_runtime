//===- FusionPass.h - Step 5: fuse matmul/conv + bias + activation ------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Pattern-matches {matmul|conv} -> [+bias] -> [relu|gelu|sigmoid] chains
// where every intermediate result has exactly one use, and rewrites the
// chain into a single `runtime.fusion_group`. A one-use requirement means
// fusion never duplicates compute — if a bias-add result feeds two
// activations, fusing into either op would recompute the add, so the
// pattern simply doesn't match.
std::unique_ptr<mlir::Pass> createFusionPass();

} // namespace runtime
