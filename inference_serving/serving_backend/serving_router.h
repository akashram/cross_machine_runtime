#pragma once
#include "../../transformer/transformer_model.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// PLAN.md Phase 9 step 8: backend-agnostic serving — the inference
// serving layer works with CPU, GPU, FPGA, and TPU backends via a
// unified router. Mirrors compiler/dialect's `DeviceKind` enum
// (CPU/GPU/FPGA/TPU/NPU/Unassigned) in spirit, but doesn't depend on it —
// that enum lives behind Phase 4's MLIR_DIR gate, and this step's router
// logic has no MLIR dependency at all.
//
// Only the CPU backend can actually execute on this Mac. GPU/FPGA/TPU/NPU
// are registered as real backend entries with an honest `available =
// false` and a reason string — same "hardware-gated but real, not
// hidden" convention this repo uses everywhere else (e.g. the "CUDA not
// found" messages root CMakeLists.txt prints) — so the router's
// fallback logic has real backends to fall back FROM, not just a
// single-entry table.
//
// NPU added 2026-07-28/30 (PLAN.md Phase 15 step 7 /
// SCOPE.md's "NPU as a fifth ServingRouter backend" note): placed last
// in the fallback priority order (see route()'s kPriorityOrder) — NPU
// toolchains (CoreML, ONNX Runtime NPU EPs) are inference-only,
// edge/mobile-first, and INT8-restricted-operator-model hardware (see
// npu_engine/op_coverage), not a general datacenter accelerator the way
// GPU/TPU/FPGA are treated here, so it's the least-preferred fallback
// rather than competing with them for priority.

namespace inference_serving {

enum class Backend { CPU, GPU, FPGA, TPU, NPU };

const char *to_string(Backend b);

struct BackendInfo {
  bool available;
  std::string unavailable_reason;  // empty if available
};

// Greedy-decodes max_new_tokens tokens continuing from `prompt`, using
// `model`. Matches transformer/transformer_test.cpp's generation loop
// exactly — the router doesn't reimplement generation, it dispatches to
// whichever backend's implementation of it.
using GenerateFn =
    std::function<std::vector<int>(const transformer::ModelParams &model, const std::vector<int> &prompt, int max_new_tokens)>;

struct RouteResult {
  Backend backend_used;
  std::vector<int> tokens;
  bool fell_back;  // true if `preferred` wasn't the backend actually used
};

class ServingRouter {
 public:
  // fn may be empty (default-constructed std::function) for an
  // unavailable backend that has no implementation to register.
  void register_backend(Backend b, BackendInfo info, GenerateFn fn = GenerateFn{});

  BackendInfo info(Backend b) const;

  // Routes to `preferred` if it's registered, available, and has a
  // generate function; otherwise falls back through a fixed priority
  // order (GPU, TPU, FPGA, CPU — fastest-to-slowest in a real deployment)
  // to the first backend that qualifies. Throws std::runtime_error if
  // none do.
  RouteResult route(Backend preferred, const transformer::ModelParams &model, const std::vector<int> &prompt,
                     int max_new_tokens) const;

 private:
  std::unordered_map<Backend, BackendInfo> infos_;
  std::unordered_map<Backend, GenerateFn> fns_;
};

// A real CPU backend: greedy-decodes via transformer::model_forward.
GenerateFn make_cpu_backend();

}  // namespace inference_serving
