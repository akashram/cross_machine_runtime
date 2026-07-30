// serving_daemon.cpp — closes the gap the Phase 16 K8s manifests
// disclosed: `serving_router.h`'s ServingRouter existed only inside a
// unit test, with no real long-running process for
// `k8s/serving/deployment.yaml` to actually point at. This is that
// process: it trains the same tiny transformer transformer_test.cpp
// validates, registers it as the CPU backend on a real ServingRouter
// (GPU/FPGA/TPU registered `available=false`, matching the router's
// existing convention), and serves real TCP connections.
//
// Wire protocol is deliberately minimal (proving the process model and
// routing work, not building a production API): one line of plain text
// per request (a prefix of the training corpus), one line of generated
// text back. `max_new_tokens` is fixed per-process via an env var since
// there's no request framing beyond a newline.
#include "serving_router.h"
#include "../../transformer/char_tokenizer.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <random>
#include <string>

using namespace inference_serving;
using namespace transformer;

namespace {

volatile sig_atomic_t g_shutdown = 0;
void on_signal(int) { g_shutdown = 1; }

int env_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::atoi(v);
}

// Same corpus/config/training loop as transformer_test.cpp's
// test_trains_and_generates(), so a known-good, already-validated recipe
// backs the daemon's model rather than a new untested one.
ModelParams train_demo_model(const CharTokenizer &tok, TransformerConfig cfg) {
  std::mt19937 init_rng(9);
  ModelParams model = init_model(cfg, init_rng);
  std::vector<int> tokens = tok.encode("the quick fox jumps ");
  constexpr int kEpochs = 400;
  constexpr float kLr = 0.05f;
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    ModelCache cache;
    Matrix logits = model_forward(model, tokens, cache);
    auto lm_loss = next_token_loss(logits, tokens);
    ModelGrads grad = zero_model_grad(cfg);
    model_backward(model, cache, lm_loss.dlogits, grad);
    sgd_step(model, grad, kLr);
  }
  return model;
}

ServingRouter make_router(const ModelParams * /*unused, CPU backend closes over the model instead*/) {
  ServingRouter router;
  router.register_backend(Backend::GPU, {false, "CUDA not found -- gpu_engine skipped"});
  router.register_backend(Backend::FPGA, {false, "XILINX_VITIS not set"});
  router.register_backend(Backend::TPU, {false, "no TPU device"});
  router.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());
  return router;
}

// Reads exactly one '\n'-terminated line (excluding the newline) from
// `fd`, or returns false on EOF/error before any newline arrives.
bool read_line(int fd, std::string &out) {
  out.clear();
  char c;
  for (;;) {
    ssize_t n = ::recv(fd, &c, 1, 0);
    if (n == 0) return !out.empty();  // EOF: treat a trailing partial line as a request
    if (n < 0) return false;
    if (c == '\n') return true;
    out.push_back(c);
  }
}

void serve_connection(int client_fd, const ServingRouter &router, const CharTokenizer &tok,
                       const ModelParams &model, int max_new_tokens, int max_seq_len) {
  std::string line;
  while (read_line(client_fd, line)) {
    if (line.empty()) continue;
    std::string response;
    try {
      std::vector<int> prompt = tok.encode(line);
      // model_forward indexes positional embeddings by absolute
      // position, sized to max_seq_len -- generating past that bound
      // trips an assert() (process abort, not a catchable exception).
      // Clamp here so one oversized request can't take the whole daemon
      // down for every other connection.
      if (static_cast<int>(prompt.size()) >= max_seq_len) {
        throw std::runtime_error("prompt too long for this model's max_seq_len");
      }
      int allowed_new_tokens = std::min(max_new_tokens, max_seq_len - static_cast<int>(prompt.size()));
      RouteResult result = router.route(Backend::CPU, model, prompt, allowed_new_tokens);
      response = tok.decode(result.tokens);
    } catch (const std::exception &e) {
      response = std::string("ERROR: ") + e.what();
    }
    response.push_back('\n');
    ssize_t sent = 0;
    while (sent < static_cast<ssize_t>(response.size())) {
      ssize_t n = ::send(client_fd, response.data() + sent, response.size() - static_cast<size_t>(sent), 0);
      if (n <= 0) { ::close(client_fd); return; }
      sent += n;
    }
  }
  ::close(client_fd);
}

}  // namespace

int main() {
  std::signal(SIGTERM, on_signal);
  std::signal(SIGINT, on_signal);

  int port = env_int("SERVING_PORT", 8080);
  int max_new_tokens = env_int("SERVING_MAX_NEW_TOKENS", 20);

  std::string corpus = "the quick fox jumps ";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                        /*max_seq_len=*/64};
  std::printf("serving_daemon: training demo model on corpus \"%s\"...\n", corpus.c_str());
  ModelParams model = train_demo_model(tok, cfg);
  ServingRouter router = make_router(&model);
  std::printf("serving_daemon: model trained, router ready (CPU available; GPU/FPGA/TPU registered unavailable)\n");

  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) { std::perror("socket"); return 1; }
  int opt = 1;
  ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    ::close(listen_fd);
    return 1;
  }
  if (::listen(listen_fd, /*backlog=*/16) < 0) {
    std::perror("listen");
    ::close(listen_fd);
    return 1;
  }
  std::printf("serving_daemon: listening on 0.0.0.0:%d (max_new_tokens=%d)\n", port, max_new_tokens);

  // Poll accept() with a timeout instead of blocking indefinitely, so a
  // SIGTERM/SIGINT (delivered any time between iterations) gets noticed
  // promptly and the socket closes cleanly instead of the process
  // getting SIGKILLed mid-accept -- the behavior a K8s pod's shutdown
  // grace period depends on.
  while (!g_shutdown) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(listen_fd, &readfds);
    timeval tv{/*tv_sec=*/1, /*tv_usec=*/0};
    int r = ::select(listen_fd + 1, &readfds, nullptr, nullptr, &tv);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (r == 0) continue;  // timeout, recheck g_shutdown
    int client_fd = ::accept(listen_fd, nullptr, nullptr);
    if (client_fd < 0) continue;
    serve_connection(client_fd, router, tok, model, max_new_tokens, cfg.max_seq_len);
  }

  std::printf("serving_daemon: shutting down cleanly\n");
  ::close(listen_fd);
  return 0;
}
