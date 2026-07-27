#include "dashboard.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace observability {

namespace {

std::string extract_str_field(const std::string &line, const std::string &key) {
  std::string needle = "\"" + key + "\":\"";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return "";
  pos += needle.size();
  auto end = line.find('"', pos);
  return line.substr(pos, end - pos);
}

double extract_num_field(const std::string &line, const std::string &key, double default_val = 0.0) {
  std::string needle = "\"" + key + "\":";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return default_val;
  pos += needle.size();
  auto end = line.find_first_of(",}", pos);
  std::string num = line.substr(pos, end - pos);
  try {
    return std::stod(num);
  } catch (...) {
    return default_val;
  }
}

}  // namespace

std::vector<MetricSample> parse_metrics_jsonl(const std::string &path) {
  std::ifstream f(path);
  std::vector<MetricSample> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    MetricSample s;
    s.backend = extract_str_field(line, "backend");
    s.rank = extract_str_field(line, "rank");
    s.metric = extract_str_field(line, "metric");
    s.value = extract_num_field(line, "value");
    s.ts_ns = static_cast<uint64_t>(extract_num_field(line, "ts_ns"));
    out.push_back(std::move(s));
  }
  return out;
}

Histogram build_histogram(const std::vector<double> &values, int num_buckets) {
  Histogram h;
  if (values.empty()) return h;
  double lo = *std::min_element(values.begin(), values.end());
  double hi = *std::max_element(values.begin(), values.end());
  if (hi <= lo) hi = lo + 1.0;  // degenerate all-equal input: one wide bucket

  h.bucket_edges.resize(static_cast<std::size_t>(num_buckets) + 1);
  h.counts.assign(static_cast<std::size_t>(num_buckets), 0);
  for (int i = 0; i <= num_buckets; ++i) h.bucket_edges[static_cast<std::size_t>(i)] = lo + (hi - lo) * i / num_buckets;

  for (double v : values) {
    int bucket = static_cast<int>((v - lo) / (hi - lo) * num_buckets);
    bucket = std::clamp(bucket, 0, num_buckets - 1);
    h.counts[static_cast<std::size_t>(bucket)]++;
  }
  return h;
}

std::string render_histogram_ascii(const Histogram &h, int bar_width) {
  if (h.counts.empty()) return "  (no data)\n";
  int max_count = *std::max_element(h.counts.begin(), h.counts.end());
  std::ostringstream oss;
  for (std::size_t i = 0; i < h.counts.size(); ++i) {
    int bar_len = (max_count > 0) ? (h.counts[i] * bar_width / max_count) : 0;
    oss << "  [" << std::fixed;
    oss.precision(2);
    oss << h.bucket_edges[i] << ".." << h.bucket_edges[i + 1] << "] ";
    for (int b = 0; b < bar_len; ++b) oss << "#";
    oss << " (" << h.counts[i] << ")\n";
  }
  return oss.str();
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  std::size_t idx = static_cast<std::size_t>(pct / 100.0 * static_cast<double>(values.size() - 1));
  return values[idx];
}

namespace {

struct Stats {
  double min_v, mean_v, max_v;
};

Stats compute_stats(const std::vector<double> &values) {
  Stats s{0, 0, 0};
  if (values.empty()) return s;
  s.min_v = *std::min_element(values.begin(), values.end());
  s.max_v = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (double v : values) sum += v;
  s.mean_v = sum / static_cast<double>(values.size());
  return s;
}

std::map<std::string, std::vector<double>> group_by(const std::vector<MetricSample> &samples, const std::string &metric_name,
                                                      bool group_by_rank) {
  std::map<std::string, std::vector<double>> groups;
  for (const auto &s : samples)
    if (s.metric == metric_name) groups[group_by_rank ? s.rank : s.backend].push_back(s.value);
  return groups;
}

}  // namespace

std::string render_report(const std::vector<MetricSample> &samples) {
  std::ostringstream oss;

  oss << "=== Latency (per backend) ===\n";
  for (auto &[backend, values] : group_by(samples, "latency_ms", /*group_by_rank=*/false)) {
    oss << "-- backend=" << backend << " (" << values.size() << " samples) --\n";
    oss << render_histogram_ascii(build_histogram(values));
    oss << "  p50=" << percentile(values, 50) << "ms  p99=" << percentile(values, 99) << "ms\n";
  }

  auto print_stats_section = [&](const char *title, const char *metric_name, const char *unit) {
    oss << "\n=== " << title << " ===\n";
    for (auto &[backend, values] : group_by(samples, metric_name, /*group_by_rank=*/false)) {
      Stats st = compute_stats(values);
      oss << "  backend=" << backend << ": min=" << st.min_v << unit << " mean=" << st.mean_v << unit
          << " max=" << st.max_v << unit << " (" << values.size() << " samples)\n";
    }
  };
  print_stats_section("GPU utilization", "gpu_util_pct", "%");
  print_stats_section("FPGA temperature", "fpga_temp_c", "C");
  print_stats_section("Collective throughput", "collective_gbps", "GB/s");

  oss << "\n=== Memory usage per rank ===\n";
  for (auto &[rank, values] : group_by(samples, "mem_gb", /*group_by_rank=*/true)) {
    Stats st = compute_stats(values);
    oss << "  rank=" << rank << ": mean=" << st.mean_v << "GB max=" << st.max_v << "GB\n";
  }

  return oss.str();
}

}  // namespace observability
