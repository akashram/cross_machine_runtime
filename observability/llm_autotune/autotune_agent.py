#!/usr/bin/env python3
"""autotune_agent.py — monitors runtime execution traces (latency per op,
device utilization, memory pressure), identifies suboptimal placement or
quantization decisions, proposes and tests changes, compares pre/post
baselines.

PLAN.md Phase 10 step 9: the final observability step, explicitly
described as coming "after all other baselines exist" — this file's
heuristic core is exactly what turns those earlier baselines
(layout_opt's alignment ceiling, hbm_sram's overlap model, gptq/kv_quant's
quantization results, the cost model) into a monitor that flags when a
*new* trace violates what those baselines already established, rather
than re-deriving bottleneck detection from scratch.

Two halves, same split as nsight_agent (step 7) and kernel_variant_agent
(step 8):
  - `detect_issues`: a real, portable, rule-based heuristic core. No LLM,
    no hardware -- pure trace analysis, actually run locally against
    synthetic-but-representative data (see README.md).
  - `propose_fix` / `run_autotuning_loop`: real Anthropic Python SDK
    client code, never invoked from this repo (same explicit decision as
    steps 7-8 -- see project memory).

Trace record shape intentionally mirrors observability/dashboard's
MetricSample (backend, rank, metric, value, ts_ns) -- the two tools are
meant to read the same trace files, dashboard for human-readable
summary, this agent for automated issue detection.
"""

import json
import sys
from dataclasses import dataclass, field

MODEL = "claude-opus-4-8"  # see SKILL.md: default unless the user names another model


@dataclass
class MetricSample:
    backend: str
    metric: str
    value: float
    rank: str = ""
    ts_ns: int = 0


@dataclass
class Issue:
    kind: str            # e.g. "placement", "quantization", "utilization"
    op_or_backend: str
    description: str
    evidence: dict = field(default_factory=dict)


def load_trace(path: str) -> list[MetricSample]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(MetricSample(
                backend=obj.get("backend", ""), metric=obj.get("metric", ""),
                value=float(obj.get("value", 0.0)), rank=obj.get("rank", ""),
                ts_ns=int(obj.get("ts_ns", 0)),
            ))
    return samples


def detect_issues(samples: list[MetricSample]) -> list[Issue]:
    """Real, portable, rule-based bottleneck detection -- no LLM, no
    hardware. Three checks, each tied to a real finding from an earlier
    step in this repo:

    1. Placement: the same op's mean latency differs sharply across
       backends (mirrors serving_bench's real finding that this repo's
       uncached CPU path runs ~9x slower per-token than a naive GPU
       estimate -- the kind of gap this check exists to catch
       automatically instead of requiring a human to notice it in a
       report).
    2. Utilization: GPU/TPU utilization sampled well below what
       layout_opt/hbm_sram's ceiling models predict is achievable at a
       given op shape signals a placement or batching problem, not a
       hardware limit.
    3. Memory pressure without quantization: high memory-pressure
       samples on a backend with no corresponding int8/int4 quantization
       metric present suggests kv_quant/gptq's real measured win (4x
       memory reduction, near-zero perplexity cost on this repo's own
       workload) isn't being applied where it could be.
    """
    issues: list[Issue] = []

    # 1. Placement: same metric name, wildly different mean by backend.
    by_metric_backend: dict[tuple[str, str], list[float]] = {}
    for s in samples:
        by_metric_backend.setdefault((s.metric, s.backend), []).append(s.value)
    metrics = {m for m, _ in by_metric_backend}
    for metric in metrics:
        backend_means = {
            backend: sum(vals) / len(vals)
            for (m, backend), vals in by_metric_backend.items() if m == metric
        }
        if len(backend_means) < 2:
            continue
        slowest_backend = max(backend_means, key=backend_means.get)
        fastest_backend = min(backend_means, key=backend_means.get)
        slow, fast = backend_means[slowest_backend], backend_means[fastest_backend]
        if fast > 0 and slow / fast >= 3.0:
            issues.append(Issue(
                kind="placement", op_or_backend=slowest_backend,
                description=(f"'{metric}' on backend '{slowest_backend}' is {slow / fast:.1f}x "
                             f"'{fast:.3g}' on '{fastest_backend}' -- consider moving this op"),
                evidence={"metric": metric, "slow_backend": slowest_backend, "slow_mean": slow,
                          "fast_backend": fastest_backend, "fast_mean": fast},
            ))

    # 2. Utilization below a conservative floor -- 50% is a placeholder
    # threshold; a real deployment would source this from the specific
    # op-shape ceiling layout_opt/hbm_sram's models compute, not a flat
    # constant. Aggregated per (metric, backend) rather than reported
    # once per sample, so a long trace doesn't drown the report in
    # near-duplicate entries.
    util_low: dict[tuple[str, str], list[float]] = {}
    for s in samples:
        if s.metric in ("gpu_util_pct", "tpu_util_pct") and s.value < 50.0:
            util_low.setdefault((s.metric, s.backend), []).append(s.value)
    for (metric, backend), vals in util_low.items():
        mean_val = sum(vals) / len(vals)
        issues.append(Issue(
            kind="utilization", op_or_backend=backend,
            description=(f"{metric} on '{backend}' averaged {mean_val:.1f}% below the 50% floor "
                         f"over {len(vals)} samples"),
            evidence={"metric": metric, "mean_value": mean_val, "num_samples": len(vals)},
        ))

    # 3. High memory pressure with no quantization metric present for that backend.
    mem_backends = {s.backend for s in samples if s.metric == "mem_pressure_pct" and s.value > 80.0}
    quant_backends = {s.backend for s in samples if s.metric in ("kv_cache_bits", "weight_bits")}
    for backend in mem_backends - quant_backends:
        issues.append(Issue(
            kind="quantization", op_or_backend=backend,
            description=(f"backend '{backend}' shows high memory pressure with no quantization metric "
                          "present -- kv_quant/gptq measured a 4x memory reduction at near-zero cost "
                          "on this repo's own workload; consider applying it here"),
            evidence={},
        ))

    return issues


def propose_fix(issues: list[Issue], trace_summary: str) -> list[dict]:
    """Sends detected issues + trace context to Claude, asks for specific
    placement/quantization change proposals. Real client code, never
    invoked -- see module docstring."""
    from anthropic import Anthropic

    client = Anthropic()
    prompt = (
        "A runtime autotuning heuristic flagged these issues in a trace:\n"
        f"{json.dumps([vars(i) for i in issues], indent=2)}\n\n"
        f"Trace summary:\n{trace_summary}\n\n"
        "For each issue, propose a specific, testable change (a placement change, "
        "a quantization to apply, a batch-size change) with your reasoning. "
        "Prioritize by expected impact."
    )
    response = client.messages.create(
        model=MODEL, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return [{"proposal": text}]  # real deployment would use output_config.format, as in nsight_agent


def run_autotuning_loop(trace_file: str, max_iterations: int = 10) -> None:
    """Full loop: ingest -> detect -> propose -> apply -> re-benchmark ->
    compare -> commit or revert. The apply/re-benchmark/compare steps are
    inherently tied to a real running system this repo doesn't have
    locally (they'd re-run whichever benchmark produced the trace, under
    the proposed change) -- documented as the hardware-gated remainder,
    not implemented as a no-op here."""
    samples = load_trace(trace_file)
    issues = detect_issues(samples)
    print(f"detected {len(issues)} issue(s) from {len(samples)} trace samples")
    for issue in issues:
        print(f"  [{issue.kind}] {issue.description}")

    if not issues:
        return
    proposals = propose_fix(issues, trace_summary=f"{len(samples)} samples, {len(issues)} issues")
    for p in proposals:
        print(p["proposal"])
    print(
        "\napply/re-benchmark/compare loop: TODO -- needs a real running "
        "system to re-benchmark the proposed change against, which this "
        "repo doesn't have locally (see README.md)."
    )


if __name__ == "__main__":
    run_autotuning_loop(sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl")
