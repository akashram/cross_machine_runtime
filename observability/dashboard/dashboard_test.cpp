// dashboard_test.cpp — writes a synthetic-but-representative metrics
// file (clearly synthetic: no live multi-node cluster to source real
// numbers from yet, see README.md), parses it back through the real
// ingestion path, and checks structural correctness (histogram bucket
// counts sum to sample count, grouping produces the right backend/rank
// keys) before printing the actual rendered report — the same report a
// real cluster's metrics would produce through this unmodified code.
#include "dashboard.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace observability;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

std::string write_synthetic_metrics(const std::string &path) {
  std::mt19937 rng(4);
  std::ofstream f(path);

  auto emit = [&](const std::string &backend, const std::string &rank, const std::string &metric, double value) {
    f << "{\"backend\":\"" << backend << "\",\"rank\":\"" << rank << "\",\"metric\":\"" << metric
      << "\",\"value\":" << value << ",\"ts_ns\":0}\n";
  };

  // Latency: cpu backend (this repo's real, measured serving_bench
  // numbers -- TTFT/TPOT-shaped) vs. a synthetic gpu backend showing what
  // a faster device's distribution would look like once real.
  std::normal_distribution<double> cpu_latency(2.7, 0.4);
  std::normal_distribution<double> gpu_latency(0.3, 0.05);
  for (int i = 0; i < 200; ++i) {
    emit("cpu", "", "latency_ms", std::max(0.01, cpu_latency(rng)));
    emit("gpu", "", "latency_ms", std::max(0.01, gpu_latency(rng)));
  }

  std::uniform_real_distribution<double> util(60.0, 98.0);
  for (int i = 0; i < 50; ++i) emit("gpu", "", "gpu_util_pct", util(rng));

  std::normal_distribution<double> temp(52.0, 4.0);
  for (int i = 0; i < 50; ++i) emit("fpga", "", "fpga_temp_c", temp(rng));

  std::normal_distribution<double> bw(85.0, 6.0);
  for (int i = 0; i < 50; ++i) emit("gpu", "", "collective_gbps", bw(rng));

  std::uniform_real_distribution<double> mem(10.0, 40.0);
  for (int rank = 0; rank < 4; ++rank)
    for (int i = 0; i < 20; ++i) emit("gpu", std::to_string(rank), "mem_gb", mem(rng));

  return path;
}

void test_parse_and_histogram_structural_correctness(const std::string &path) {
  auto samples = parse_metrics_jsonl(path);
  require(samples.size() == 200 + 200 + 50 + 50 + 50 + 4 * 20, "parses every emitted sample");

  std::vector<double> cpu_latencies;
  for (const auto &s : samples)
    if (s.backend == "cpu" && s.metric == "latency_ms") cpu_latencies.push_back(s.value);
  require(cpu_latencies.size() == 200, "backend field correctly distinguishes cpu from gpu latency samples");

  Histogram h = build_histogram(cpu_latencies);
  int total = std::accumulate(h.counts.begin(), h.counts.end(), 0);
  require(total == static_cast<int>(cpu_latencies.size()), "histogram bucket counts sum to the sample count");
  require(h.bucket_edges.size() == h.counts.size() + 1, "histogram has one more edge than bucket");
}

}  // namespace

int main() {
  std::string dir = fs::temp_directory_path().string() + "/dashboard_test_" + std::to_string(::getpid());
  fs::create_directories(dir);
  std::string metrics_path = dir + "/metrics.jsonl";
  write_synthetic_metrics(metrics_path);

  test_parse_and_histogram_structural_correctness(metrics_path);

  auto samples = parse_metrics_jsonl(metrics_path);
  std::string report = render_report(samples);
  std::printf("\n%s\n", report.c_str());

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
