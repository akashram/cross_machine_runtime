// training_worker.cpp — closes the second gap the Phase 16 fork
// disclosed: `k8s/training/statefulset.yaml` needs one real OS PROCESS
// per rank/pod, but every existing multi-rank step in this repo
// (canonically `full_training_loop/training_loop_test.cpp`) simulates
// ranks as std::async tasks inside ONE process, over a loopback TCP mesh
// all ranks share a single hostname for. This file is a real
// process-per-rank driver: it reads RANK/WORLD_SIZE/PEER_HOSTS from the
// environment (the standard convention a Kubernetes StatefulSet's own
// entrypoint script would set, one hostname per rank -- see
// networking/common/channel.h's new peer_hosts constructor, added
// alongside this file), builds exactly ONE TcpChannel for THIS rank, and
// runs the identical forward -> backward -> clip -> grad-sync -> ZeRO-1
// step -> checkpoint loop training_loop_test.cpp's per-rank lambda body
// runs -- same model, same optimizer, same collective calls, just across
// real inter-process (not intra-process) TCP this time.
//
// Deliberately NOT a refactor of training_loop_test.cpp: that file stays
// untouched (it's an existing passing test), and this duplicates only the
// ~15-line toy-dataset generator, not any of the actual training math —
// the model/optimizer/clipping/checkpoint classes are the SAME headers,
// reused unchanged.
#include "../autograd/mlp.h"
#include "../grad_clipping/grad_clipping.h"
#include "../checkpoint/sharded_checkpoint.h"
#include "../zero1/zero1_optimizer.h"
#include "../../networking/common/channel.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace distributed_training;
namespace fs = std::filesystem;

namespace {

struct ToyDataset {
  Matrix x;
  std::vector<int> labels;
  int total;
};

// Deterministic given the seed -- every rank/process independently
// regenerates the SAME full dataset (and, separately, the SAME initial
// model) this way, rather than one process computing it and shipping it
// to the others. That's what lets every rank agree on shard boundaries
// and starting weights with zero extra coordination messages.
ToyDataset make_toy_dataset(std::mt19937 &rng) {
  constexpr int kClasses = 3;
  constexpr int kPerClass = 40;
  ToyDataset ds{Matrix(kClasses * kPerClass, 2), {}, kClasses * kPerClass};
  ds.labels.resize(static_cast<size_t>(ds.total));
  std::normal_distribution<float> noise(0.0f, 0.3f);
  const float centers[kClasses][2] = {{0.0f, 0.0f}, {3.0f, 3.0f}, {-3.0f, 3.0f}};
  int idx = 0;
  for (int c = 0; c < kClasses; ++c) {
    for (int s = 0; s < kPerClass; ++s) {
      ds.x(idx, 0) = centers[c][0] + noise(rng);
      ds.x(idx, 1) = centers[c][1] + noise(rng);
      ds.labels[static_cast<size_t>(idx)] = c;
      ++idx;
    }
  }
  return ds;
}

Matrix row_slice(const Matrix &m, int start, int count) {
  Matrix out(count, m.cols());
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}

float compute_loss(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  Tensor xt(x);
  return softmax_cross_entropy(mlp.forward(xt), labels).value()(0, 0);
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

std::string require_env(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !*v) throw std::runtime_error(std::string("training_worker: required env var ") + name + " not set");
  return v;
}

int env_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::atoi(v);
}

float env_float(const char *name, float fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::strtof(v, nullptr);
}

std::vector<std::string> split_commas(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

}  // namespace

int main() {
  const int rank = std::atoi(require_env("RANK").c_str());
  const int world_size = std::atoi(require_env("WORLD_SIZE").c_str());
  std::vector<std::string> peer_hosts = split_commas(require_env("PEER_HOSTS"));
  if (static_cast<int>(peer_hosts.size()) != world_size) {
    throw std::runtime_error("training_worker: PEER_HOSTS must list exactly WORLD_SIZE hostnames");
  }
  const auto base_port = static_cast<uint16_t>(env_int("BASE_PORT", 46201));
  const int epochs = env_int("EPOCHS", 30);
  const float lr = env_float("LR", 0.05f);
  const float max_grad_norm = env_float("MAX_GRAD_NORM", 5.0f);
  const int checkpoint_every = env_int("CHECKPOINT_EVERY", 10);
  const std::string ckpt_dir = [] {
    const char *v = std::getenv("CHECKPOINT_DIR");
    return v && *v ? std::string(v) : fs::temp_directory_path().string() + "/training_worker_default";
  }();
  fs::create_directories(ckpt_dir);

  std::printf("training_worker: rank=%d world_size=%d base_port=%u epochs=%d ckpt_dir=%s\n", rank, world_size,
              base_port, epochs, ckpt_dir.c_str());

  // Real inter-process TCP handshake: this rank binds 0.0.0.0:base_port+rank,
  // accepts from every higher rank, and dials every lower rank at its own
  // PEER_HOSTS entry -- the constructor blocks until the full mesh (across
  // however many separate OS processes are participating) is connected.
  netcommon::TcpChannel channel(rank, world_size, base_port, peer_hosts);
  std::printf("rank %d: connected to all %d peers over real TCP\n", rank, world_size);

  std::mt19937 data_rng(99);
  ToyDataset ds = make_toy_dataset(data_rng);
  int shard_size = ds.total / world_size;

  std::mt19937 init_rng(42);
  MLP init_mlp({2, 16, 3}, init_rng);
  std::vector<float> initial_flat = flatten_params(init_mlp.parameters());

  std::mt19937 rank_rng(1000 + rank);
  MLP mlp({2, 16, 3}, rank_rng);
  auto params = mlp.parameters();
  unflatten_params(params, initial_flat);
  size_t total = total_param_count(params);

  ZeroStage1Optimizer opt(total, rank, world_size, lr);
  std::vector<float> full_params = initial_flat;
  full_params.resize(opt.padded_total_params(), 0.0f);

  Matrix shard_x = row_slice(ds.x, rank * shard_size, shard_size);
  std::vector<int> shard_labels(ds.labels.begin() + rank * shard_size, ds.labels.begin() + (rank + 1) * shard_size);

  AsyncCheckpointWriter ckpt_writer;
  std::map<std::string, double> phase_ms{{"forward_backward", 0}, {"grad_clip", 0}, {"grad_sync", 0},
                                          {"optimizer_step", 0}, {"checkpoint", 0}};
  float first_loss = 0.0f, last_loss = 0.0f;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    unflatten_params(params, full_params);
    if (rank == 0) {
      float l = compute_loss(mlp, ds.x, ds.labels);
      if (epoch == 0) first_loss = l;
      last_loss = l;
    }

    auto t0 = Clock::now();
    Tensor x(shard_x);
    Tensor logits = mlp.forward(x);
    Tensor loss = softmax_cross_entropy(logits, shard_labels);
    zero_grad(params);
    loss.backward();
    phase_ms["forward_backward"] += ms_since(t0);

    auto grad = flatten_grads(params);
    grad.resize(opt.padded_total_params(), 0.0f);

    auto t1 = Clock::now();
    float gnorm = global_grad_norm(grad, channel);
    clip_grad_by_global_norm(grad, gnorm, max_grad_norm);
    phase_ms["grad_clip"] += ms_since(t1);

    auto t2 = Clock::now();
    ring_allreduce(grad.data(), grad.size(), channel);
    for (float &g : grad) g /= static_cast<float>(world_size);
    phase_ms["grad_sync"] += ms_since(t2);

    auto t3 = Clock::now();
    opt.step(full_params, grad, channel);
    phase_ms["optimizer_step"] += ms_since(t3);

    if ((epoch + 1) % checkpoint_every == 0) {
      auto t4 = Clock::now();
      ckpt_writer.wait();
      std::vector<float> shard(full_params.begin() + static_cast<long>(static_cast<size_t>(rank) * opt.shard_size()),
                                full_params.begin() + static_cast<long>(static_cast<size_t>(rank + 1) * opt.shard_size()));
      ckpt_writer.start_write(ckpt_dir + "/rank" + std::to_string(rank) + "_epoch" + std::to_string(epoch) + ".bin", shard);
      phase_ms["checkpoint"] += ms_since(t4);
    }
  }
  ckpt_writer.wait();

  if (rank == 0) {
    std::printf("rank 0: training loss %.4f -> %.4f\n", first_loss, last_loss);
  }

  std::vector<std::string> phase_names{"forward_backward", "grad_clip", "grad_sync", "optimizer_step", "checkpoint"};
  double total_ms = 0.0;
  for (auto &name : phase_names) total_ms += phase_ms[name];
  std::printf("rank %d phase breakdown (%d steps): ", rank, epochs);
  for (auto &name : phase_names) std::printf("%s=%.3fms ", name.c_str(), phase_ms[name]);
  std::printf("total=%.3fms\n", total_ms);

  std::printf("rank %d: PASS\n", rank);
  return 0;
}
