#pragma once
#include <cstdint>
#include <string>
#include <vector>

// PLAN.md Phase 10 step 3: unified observability dashboard — CLI report
// combining latency histograms per backend, GPU utilization, FPGA
// temperature, collective throughput, memory usage per rank.
//
// Ingests a flat, line-oriented JSON metrics format this repo controls
// (one object per line: backend, rank, metric name, value, timestamp) —
// the natural sink for observability/opentelemetry's spans (step 2,
// which already emits comparable per-line JSON) and observability/ebpf's
// trace events (step 1), once both run on real hardware and are
// converted to this shape. `parse_metrics_jsonl`'s hand-rolled scanner
// deliberately only handles this repo's own flat (no nesting) emission
// format — not a general JSON parser — matching the "hand-roll what's
// actually needed, verify it" pattern `opentelemetry`'s Design section
// documents for the same reason.

namespace observability {

struct MetricSample {
  std::string backend;  // e.g. "cpu", "gpu", "fpga", "tpu"
  std::string rank;     // e.g. "0", "1" -- empty if not rank-scoped
  std::string metric;   // e.g. "latency_ms", "gpu_util_pct", "fpga_temp_c", "collective_gbps", "mem_gb"
  double value;
  uint64_t ts_ns;
};

std::vector<MetricSample> parse_metrics_jsonl(const std::string &path);

struct Histogram {
  std::vector<double> bucket_edges;  // num_buckets + 1 edges
  std::vector<int> counts;           // num_buckets counts
};

Histogram build_histogram(const std::vector<double> &values, int num_buckets = 10);
std::string render_histogram_ascii(const Histogram &h, int bar_width = 30);

double percentile(std::vector<double> values, double pct);  // takes by value: sorts internally

// Renders the full unified report: a "Latency (per backend)" section
// (histogram + p50/p99 per backend, from the "latency_ms" metric), a
// "GPU utilization" / "FPGA temperature" / "collective throughput"
// section (min/mean/max per backend, from "gpu_util_pct"/"fpga_temp_c"/
// "collective_gbps"), and a "Memory usage per rank" section (mean/max
// grouped by rank, from "mem_gb").
std::string render_report(const std::vector<MetricSample> &samples);

}  // namespace observability
