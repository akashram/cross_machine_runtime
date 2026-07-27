#include "serving_router.h"

namespace inference_serving {

const char *to_string(Backend b) {
  switch (b) {
    case Backend::CPU: return "cpu";
    case Backend::GPU: return "gpu";
    case Backend::FPGA: return "fpga";
    case Backend::TPU: return "tpu";
  }
  return "unknown";
}

void ServingRouter::register_backend(Backend b, BackendInfo info, GenerateFn fn) {
  infos_[b] = info;
  if (fn) fns_[b] = std::move(fn);
}

BackendInfo ServingRouter::info(Backend b) const {
  auto it = infos_.find(b);
  if (it == infos_.end()) return BackendInfo{false, "backend not registered"};
  return it->second;
}

namespace {
bool qualifies(const std::unordered_map<Backend, BackendInfo> &infos, const std::unordered_map<Backend, GenerateFn> &fns,
               Backend b) {
  auto info_it = infos.find(b);
  if (info_it == infos.end() || !info_it->second.available) return false;
  return fns.find(b) != fns.end();
}
}  // namespace

RouteResult ServingRouter::route(Backend preferred, const transformer::ModelParams &model, const std::vector<int> &prompt,
                                  int max_new_tokens) const {
  static const Backend kPriorityOrder[] = {Backend::GPU, Backend::TPU, Backend::FPGA, Backend::CPU};

  Backend chosen = preferred;
  bool fell_back = false;
  if (!qualifies(infos_, fns_, preferred)) {
    fell_back = true;
    bool found = false;
    for (Backend b : kPriorityOrder) {
      if (qualifies(infos_, fns_, b)) {
        chosen = b;
        found = true;
        break;
      }
    }
    if (!found)
      throw std::runtime_error("ServingRouter::route: no available backend with a registered generate function");
  }

  RouteResult result;
  result.backend_used = chosen;
  result.fell_back = fell_back;
  result.tokens = fns_.at(chosen)(model, prompt, max_new_tokens);
  return result;
}

GenerateFn make_cpu_backend() {
  return [](const transformer::ModelParams &model, const std::vector<int> &prompt, int max_new_tokens) {
    std::vector<int> generated = prompt;
    for (int i = 0; i < max_new_tokens; ++i) {
      transformer::ModelCache cache;
      transformer::Matrix logits = transformer::model_forward(model, generated, cache);
      int last = static_cast<int>(generated.size()) - 1;
      int argmax = 0;
      float best = logits(last, 0);
      for (int v = 1; v < model.config.vocab_size; ++v)
        if (logits(last, v) > best) { best = logits(last, v); argmax = v; }
      generated.push_back(argmax);
    }
    return generated;
  };
}

}  // namespace inference_serving
