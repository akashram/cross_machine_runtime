//===- AotCompiler.h - Step 12: end-to-end compilation pipeline ---------===//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace runtime {

struct CompilerOptions {
  int num_shard_devices = 1;
  int shard_strategy = 0;       // ShardingStrategy, as int to keep this header MLIR-free
  unsigned affine_tile_size = 32;
  double remat_threshold = 8.0;
  std::vector<std::string> link_libraries; // e.g. cpu_engine.a, gpu_engine.a paths
  bool verbose = false;
};

// Wall-clock time per pipeline stage, in milliseconds — fills the
// "Results" table in compiler/aot/README.md once run on Linux.
struct CompileStats {
  double parse_ms = 0;
  double shape_inference_ms = 0;
  double fusion_ms = 0;
  double affine_lower_ms = 0;
  double mem_planning_ms = 0;
  double remat_ms = 0;
  double placement_ms = 0;
  double sharding_ms = 0;
  double kernel_spec_ms = 0;
  double codegen_ms = 0; // LLVM IR translation + object emission + link
  double total_ms = 0;
  int64_t kernels_lowered = 0;
  int64_t kernels_skipped = 0;
};

// Orchestrates every Phase 4 pass end-to-end: parse -> shape inference ->
// fusion -> affine lowering/tiling -> memory planning -> rematerialization
// -> placement -> sharding -> kernel specialization -> standard MLIR
// lowering to LLVM dialect -> LLVM IR -> object file -> link against
// `link_libraries` into a native binary. Each stage's wall-clock time is
// recorded in a CompileStats returned by compile().
class AotCompiler {
public:
  explicit AotCompiler(CompilerOptions opts);

  // Returns false and leaves `output_binary_path` unwritten if any stage
  // fails (e.g. the parsed IR doesn't verify); check getStats() either way
  // for the timing breakdown up to the failure point.
  bool compile(const std::string &input_ir_path,
               const std::string &output_binary_path);

  const CompileStats &getStats() const { return stats_; }

private:
  CompilerOptions opts_;
  CompileStats stats_;
};

} // namespace runtime
