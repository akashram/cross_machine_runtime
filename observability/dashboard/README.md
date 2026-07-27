# dashboard

**Status: code-complete AND locally run — no hardware dependency for the
ingestion/aggregation/rendering pipeline itself; the input data below is
clearly-labeled synthetic (no live multi-node cluster to source real
metrics from yet), not a claim of real measurement.**

## What this measures

PLAN.md Phase 10 step 3: unified CLI report combining latency histograms
per backend, GPU utilization, FPGA temperature, collective throughput,
memory usage per rank.

## Design

- `parse_metrics_jsonl`: reads a flat, line-oriented JSON metrics format
  this repo controls (`backend`, `rank`, `metric`, `value`, `ts_ns` per
  line) — the natural sink for `observability/opentelemetry`'s spans
  (step 2, comparable per-line JSON already) and `observability/ebpf`'s
  trace events (step 1), once both run on real hardware and are converted
  to this shape. Hand-rolled field extraction, not a general JSON parser
  — matches the scope `opentelemetry/README.md` documents for the same
  reason: this repo's own emitters produce flat (no nesting) JSON, and
  that's all this needs to handle correctly.
- `build_histogram`/`render_histogram_ascii`: real histogram bucketing +
  ASCII bar rendering, verified structurally in `dashboard_test.cpp`
  (bucket counts sum to sample count).
- `render_report`: five sections — latency histogram + p50/p99 per
  backend, then min/mean/max for GPU utilization / FPGA temperature /
  collective throughput per backend, then mean/max memory per rank.
- `dashboard_report`: the actual CLI binary (`dashboard_report
  metrics1.jsonl [metrics2.jsonl ...]`) — usable standalone against real
  metrics files once they exist, not just exercised through the test.
- `dashboard_test.cpp` generates synthetic-but-representative input (a
  cpu-backend latency distribution shaped like `serving_bench`'s real
  TTFT/TPOT numbers, a faster synthetic gpu-backend distribution, and
  representative GPU util / FPGA temp / collective bandwidth / per-rank
  memory data) — clearly synthetic, since there's no live cluster to
  source real numbers from — then runs it through the real
  parse-aggregate-render pipeline unmodified. The report below is
  real, unedited output; only the input feeding it is synthetic.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
=== Latency (per backend) ===
-- backend=cpu (200 samples) --
  [1.84..2.04] ###### (9)
  [2.04..2.23] ########## (14)
  [2.23..2.43] ################# (24)
  [2.43..2.63] ############################## (42)
  [2.63..2.82] ##################### (30)
  [2.82..3.02] ############################ (40)
  [3.02..3.22] ################# (25)
  [3.22..3.41] ######### (13)
  [3.41..3.61]  (1)
  [3.61..3.81] # (2)
  p50=2.68637ms  p99=3.44209ms
-- backend=gpu (200 samples) --
  [0.16..0.19] # (4)
  [0.19..0.22]  (2)
  [0.22..0.24] ####### (16)
  [0.24..0.27] ########## (22)
  [0.27..0.30] ############################## (61)
  [0.30..0.33] ##################### (43)
  [0.33..0.36] ################### (40)
  [0.36..0.39] ### (8)
  [0.39..0.42] # (3)
  [0.42..0.45]  (1)
  p50=0.300222ms  p99=0.41004ms

=== GPU utilization ===
  backend=gpu: min=60.8046% mean=78.1912% max=97.5027% (50 samples)

=== FPGA temperature ===
  backend=fpga: min=47.5554C mean=53.135C max=61.6486C (50 samples)

=== Collective throughput ===
  backend=gpu: min=73.0299GB/s mean=86.9598GB/s max=103.672GB/s (50 samples)

=== Memory usage per rank ===
  rank=0: mean=23.6839GB max=37.7112GB
  rank=1: mean=27.4155GB max=39.9275GB
  rank=2: mean=29.0703GB max=39.8908GB
  rank=3: mean=25.3429GB max=39.4612GB
```

## Hardware notes
None for the tool itself. Real input requires either a live multi-node/
multi-backend run emitting metrics in this format, or a conversion step
from `observability/ebpf`'s and `observability/opentelemetry`'s real
trace output (both themselves hardware-gated) into `MetricSample`
records.
