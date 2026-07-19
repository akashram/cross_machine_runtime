// aotc_main.cpp — CLI driver for AotCompiler. Prints a stage-by-stage
// timing table matching compiler/aot/README.md's Results table, so
// running this on Linux and pasting the output is the whole "fill in the
// README" step.
//
// Usage: runtime_aotc <input.mlir> <output_binary> [--devices N]
//                      [--tile-size N] [--remat-threshold F] [--verbose]
//                      [--link lib.a ...]

#include "AotCompiler.h"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <input.mlir> <output_binary> [options]\n", argv[0]);
    return 2;
  }

  runtime::CompilerOptions opts;
  std::string input = argv[1];
  std::string output = argv[2];

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (arg == "--devices") opts.num_shard_devices = std::stoi(next());
    else if (arg == "--tile-size") opts.affine_tile_size = static_cast<unsigned>(std::stoi(next()));
    else if (arg == "--remat-threshold") opts.remat_threshold = std::stod(next());
    else if (arg == "--verbose") opts.verbose = true;
    else if (arg == "--link") opts.link_libraries.push_back(next());
    else std::fprintf(stderr, "warning: unrecognized option '%s'\n", arg.c_str());
  }

  runtime::AotCompiler compiler(opts);
  bool ok = compiler.compile(input, output);
  const runtime::CompileStats &s = compiler.getStats();

  std::printf("%-24s %10s\n", "stage", "time (ms)");
  std::printf("%-24s %10.3f\n", "parse", s.parse_ms);
  std::printf("%-24s %10.3f\n", "shape_inference", s.shape_inference_ms);
  std::printf("%-24s %10.3f\n", "fusion", s.fusion_ms);
  std::printf("%-24s %10.3f\n", "affine_lower", s.affine_lower_ms);
  std::printf("%-24s %10.3f\n", "mem_planning", s.mem_planning_ms);
  std::printf("%-24s %10.3f\n", "remat", s.remat_ms);
  std::printf("%-24s %10.3f\n", "placement", s.placement_ms);
  std::printf("%-24s %10.3f\n", "sharding", s.sharding_ms);
  std::printf("%-24s %10.3f\n", "kernel_spec", s.kernel_spec_ms);
  std::printf("%-24s %10.3f\n", "codegen", s.codegen_ms);
  std::printf("%-24s %10.3f\n", "total", s.total_ms);
  std::printf("kernels lowered: %lld, skipped: %lld\n",
              static_cast<long long>(s.kernels_lowered),
              static_cast<long long>(s.kernels_skipped));

  if (!ok) {
    std::fprintf(stderr, "compilation failed — see stage timings above for how far it got\n");
    return 1;
  }
  return 0;
}
