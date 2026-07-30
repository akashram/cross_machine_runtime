// serving_router_test.cpp — router logic tests (fallback priority,
// throwing when nothing qualifies) plus one real end-to-end check: routed
// through GPU (unavailable) -> falls back to the real CPU backend ->
// produces the exact same greedy-decoded tokens transformer/'s own
// generation loop would, using a real trained model.
#include "serving_router.h"
#include "../../transformer/char_tokenizer.h"

#include <cstdio>
#include <random>

using namespace inference_serving;
using namespace transformer;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

ServingRouter make_router_cpu_only() {
  ServingRouter router;
  router.register_backend(Backend::GPU, {false, "CUDA not found — gpu_engine skipped"});
  router.register_backend(Backend::FPGA, {false, "XILINX_VITIS not set"});
  router.register_backend(Backend::TPU, {false, "no TPU device"});
  router.register_backend(Backend::NPU, {false, "no NPU/ANE toolchain (coremltools/onnxruntime not installed)"});
  router.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());
  return router;
}

void test_falls_back_to_only_available_backend() {
  ServingRouter router = make_router_cpu_only();
  std::mt19937 rng(1);
  TransformerConfig cfg{4, 4, 2, 1, 8, 8};
  ModelParams model = init_model(cfg, rng);

  RouteResult result = router.route(Backend::GPU, model, {0}, /*max_new_tokens=*/3);
  require(result.backend_used == Backend::CPU, "falls back to CPU when GPU is unavailable");
  require(result.fell_back, "reports that a fallback occurred");
  require(result.tokens.size() == 4, "generated prompt + max_new_tokens tokens");
}

void test_no_fallback_when_preferred_is_available() {
  ServingRouter router = make_router_cpu_only();
  std::mt19937 rng(1);
  TransformerConfig cfg{4, 4, 2, 1, 8, 8};
  ModelParams model = init_model(cfg, rng);

  RouteResult result = router.route(Backend::CPU, model, {0}, 2);
  require(result.backend_used == Backend::CPU, "uses the preferred backend directly");
  require(!result.fell_back, "reports no fallback when the preferred backend qualifies");
}

void test_priority_order_prefers_higher_priority_fallback() {
  ServingRouter router;
  router.register_backend(Backend::FPGA, {false, "unavailable"});
  // TPU is available with a stub generator; CPU is also available. Per
  // the fixed priority order (GPU, TPU, FPGA, CPU), a request for the
  // unavailable FPGA backend should fall back to TPU, not CPU, since TPU
  // outranks CPU in the priority list even though both qualify.
  router.register_backend(Backend::TPU, {true, ""}, [](const ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-1};  // marker value identifying "TPU ran"
  });
  router.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());

  std::mt19937 rng(1);
  TransformerConfig cfg{4, 4, 2, 1, 8, 8};
  ModelParams model = init_model(cfg, rng);
  RouteResult result = router.route(Backend::FPGA, model, {0}, 2);
  require(result.backend_used == Backend::TPU, "prefers TPU over CPU as the fallback, per priority order");
  require(result.tokens == std::vector<int>{-1}, "actually dispatched to TPU's generate function, not CPU's");
}

void test_npu_outranks_cpu_but_not_fpga_in_fallback() {
  // Per the fixed priority order (GPU, TPU, FPGA, NPU, CPU), an available
  // NPU should be preferred over CPU but should lose to an available FPGA.
  ServingRouter router;
  router.register_backend(Backend::NPU, {true, ""}, [](const ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-2};  // marker value identifying "NPU ran"
  });
  router.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());

  std::mt19937 rng(1);
  TransformerConfig cfg{4, 4, 2, 1, 8, 8};
  ModelParams model = init_model(cfg, rng);
  RouteResult result = router.route(Backend::GPU, model, {0}, 2);
  require(result.backend_used == Backend::NPU, "prefers NPU over CPU as the fallback, per priority order");
  require(result.tokens == std::vector<int>{-2}, "actually dispatched to NPU's generate function, not CPU's");

  router.register_backend(Backend::FPGA, {true, ""}, [](const ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-3};  // marker value identifying "FPGA ran"
  });
  RouteResult result2 = router.route(Backend::GPU, model, {0}, 2);
  require(result2.backend_used == Backend::FPGA, "prefers FPGA over NPU as the fallback, per priority order");
}

void test_throws_when_nothing_available() {
  ServingRouter router;
  router.register_backend(Backend::GPU, {false, "no CUDA"});
  std::mt19937 rng(1);
  TransformerConfig cfg{4, 4, 2, 1, 8, 8};
  ModelParams model = init_model(cfg, rng);

  bool threw = false;
  try {
    router.route(Backend::GPU, model, {0}, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "throws when no registered backend is both available and has a generate function");
}

void test_cpu_backend_matches_direct_greedy_generation() {
  std::string corpus = "ab";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), 8, 2, 1, 16, 8};
  std::mt19937 rng(9);
  ModelParams model = init_model(cfg, rng);

  ServingRouter router = make_router_cpu_only();
  RouteResult routed = router.route(Backend::CPU, model, {0}, 3);

  // Direct greedy generation, bypassing the router entirely.
  std::vector<int> direct{0};
  for (int i = 0; i < 3; ++i) {
    ModelCache cache;
    Matrix logits = model_forward(model, direct, cache);
    int last = static_cast<int>(direct.size()) - 1;
    int argmax = 0;
    float best = logits(last, 0);
    for (int v = 1; v < cfg.vocab_size; ++v)
      if (logits(last, v) > best) { best = logits(last, v); argmax = v; }
    direct.push_back(argmax);
  }
  require(routed.tokens == direct, "router's CPU backend produces identical output to direct greedy generation");
}

} // namespace

int main() {
  test_falls_back_to_only_available_backend();
  test_no_fallback_when_preferred_is_available();
  test_priority_order_prefers_higher_priority_fallback();
  test_npu_outranks_cpu_but_not_fpga_in_fallback();
  test_throws_when_nothing_available();
  test_cpu_backend_matches_direct_greedy_generation();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
