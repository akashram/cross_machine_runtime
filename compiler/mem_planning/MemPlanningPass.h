//===- MemPlanningPass.h - Step 7: buffer liveness + reuse planning -----===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Computes a static buffer plan for every statically-shaped tensor value
// in a function: a byte offset into one shared arena, chosen so that two
// values with overlapping live ranges never share bytes, and values with
// disjoint live ranges are free to reuse the same bytes. Does not rewrite
// IR — it *annotates* each op's result with a `runtime.buffer_offset`
// attribute and the function with `runtime.peak_memory_bytes`, so the AOT
// pipeline (step 12) and kernel specialization (step 11) can consume the
// plan without this pass needing to know how allocation is eventually
// realized (arena mmap, memref.alloc, whatever the target backend wants).
std::unique_ptr<mlir::Pass> createMemoryPlanningPass();

} // namespace runtime
