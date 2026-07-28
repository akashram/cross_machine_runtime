# nsight_agent

**Status: portable half (the `ncu --csv` parser) code-complete AND locally
run; the Claude API call is real client code, never invoked — see Design.**

## What this measures

PLAN.md Phase 10 step 7: ingests Nsight Compute profiles, parses key
metrics (SM utilization, memory bandwidth, warp efficiency), outputs a
ranked list of optimization suggestions via the Claude API. Runs
automatically in CI on benchmark PRs.

## Design

- `parse_ncu_metrics`: a real parser for `ncu --csv` output's actual shape
  (row 0 = metric names, row 1 = units, one row per kernel invocation
  after that) — no GPU needed to exercise this against a sample CSV, and
  it's actually run and verified against one below.
- `analyze_nsight_profile`: real Anthropic Python SDK usage —
  `client.messages.create(model="claude-opus-4-8", ...)` with
  `output_config.format` (structured outputs, JSON-schema-validated) so
  the response is a typed list of `{priority, kernel, metric, suggestion,
  expected_gain}` objects, not free-text JSON hand-parsed out of a
  response — the current recommended pattern per this repo's Claude API
  reference, not the `response.content[0].text` + manual `json.loads`
  the original stub sketched.
- **Deliberately never invoked**: per an explicit decision this session
  (see project memory), this repo's three AI-agent steps (this one, step
  8, step 9) get real, complete client code but no actual API calls from
  within this build — calling out would spend this session's API budget
  on a benchmark-agent side task, not the conversation's direct scope.
  Treated as gated the same way GPU/TPU hardware steps are, just gated on
  "an Nsight profile + an API key" instead of a GPU.

## Results (captured 2026-07-27, this Mac — parser only)

```
[
  {
    "kernel_name": "matmul_kernel(float*, float*, float*)",
    "metrics": {
      "sm__throughput.avg.pct_of_peak_sustained_elapsed": {"value": "42.3", "unit": "%"},
      "dram__throughput.avg.pct_of_peak_sustained_elapsed": {"value": "88.1", "unit": "%"}
    }
  },
  {
    "kernel_name": "relu_kernel(float*)",
    "metrics": {
      "sm__throughput.avg.pct_of_peak_sustained_elapsed": {"value": "95.0", "unit": "%"},
      "dram__throughput.avg.pct_of_peak_sustained_elapsed": {"value": "12.4", "unit": "%"}
    }
  }
]
```
Real output from `parse_ncu_metrics` against a sample two-header-row `ncu
--csv`-shaped input — confirms the parser handles the actual Nsight
Compute export format, independent of whether Claude ever sees it.

## Hardware notes
- Required for the full pipeline: an NVIDIA GPU (to produce a real
  `.ncu-rep`/CSV export) and an `ANTHROPIC_API_KEY` (or `ant auth login`
  profile) to actually call `analyze_nsight_profile`.
