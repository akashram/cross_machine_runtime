//===- RematPass.h - Step 8: rematerialization ---------------------------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Single-op-granularity rematerialization: for a cheap-to-recompute op
// (elementwise: relu/gelu/sigmoid/add/mul/sub) whose result is kept alive
// for more than `remat_threshold` ops between its definition and its
// farthest use, clone the op at every use site past the first instead of
// keeping the original result resident that whole span. Shrinks the
// original's live range to [def, nearest use] for the memory planning pass
// (step 7) to then place a smaller interval.
//
// Deliberately does NOT do segment-level activation checkpointing (the
// "recompute the whole forward pass during backward" scheme from the
// training literature) — that requires knowing the forward/backward
// boundary, which this generic dialect-level pass has no signal for. This
// is the mechanical building block; a training-loop-aware pass in Phase 6
// can call this repeatedly with a segment boundary once that context
// exists.
std::unique_ptr<mlir::Pass> createRematerializationPass(double remat_threshold);

} // namespace runtime
