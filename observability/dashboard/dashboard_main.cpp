// dashboard_main.cpp — CLI entry point: `dashboard_report metrics1.jsonl
// [metrics2.jsonl ...]` prints the unified report across all given files.
// The metrics files this ingests are what observability/ebpf (step 1) and
// observability/opentelemetry (step 2) would produce once converted to
// this repo's flat MetricSample JSON shape on real hardware.
#include "dashboard.h"

#include <cstdio>

using namespace observability;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s metrics1.jsonl [metrics2.jsonl ...]\n", argv[0]);
    return 1;
  }

  std::vector<MetricSample> all_samples;
  for (int i = 1; i < argc; ++i) {
    auto samples = parse_metrics_jsonl(argv[i]);
    all_samples.insert(all_samples.end(), samples.begin(), samples.end());
  }

  std::printf("%s\n", render_report(all_samples).c_str());
  return 0;
}
