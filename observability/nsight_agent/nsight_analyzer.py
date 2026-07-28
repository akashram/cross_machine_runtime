#!/usr/bin/env python3
"""nsight_analyzer.py — ingests Nsight Systems/Compute profile metrics and
asks Claude to produce a ranked list of optimization suggestions.

PLAN.md Phase 10 step 7: "ingests Nsight Systems `.nsys-rep` or Nsight
Compute `.ncu-rep`, parses key metrics (SM utilization, memory bandwidth,
warp efficiency), outputs ranked list of optimization suggestions with
specific guidance. Runs automatically in CI on benchmark PRs."

Real client code, using the current Anthropic Python SDK
(`client.messages.create`, structured outputs via `output_config.format`
rather than hand-parsing free-text JSON out of a response). No API calls
are made from this repo — see README.md for why (this step is treated as
hardware/API-gated, same as GPU-hardware steps, rather than actually
invoked).

`parse_ncu_metrics` is real and portable: `ncu --csv` output has two
header rows (metric names, then units) followed by one row per kernel
invocation — this parses that real format, no GPU needed to exercise the
parser itself against a sample CSV.
"""

import csv
import io
import json
import sys

MODEL = "claude-opus-4-8"  # see SKILL.md: default unless the user names another model

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "integer", "description": "1 = highest priority"},
                    "kernel": {"type": "string"},
                    "metric": {"type": "string", "description": "the metric that motivates this suggestion"},
                    "suggestion": {"type": "string", "description": "specific, actionable guidance"},
                    "expected_gain": {"type": "string", "description": "e.g. '~15% latency reduction'"},
                },
                "required": ["priority", "kernel", "metric", "suggestion", "expected_gain"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}


def parse_ncu_metrics(csv_text: str) -> list[dict]:
    """Parses real `ncu --csv` output: row 0 is metric names, row 1 is
    units, every row after is one kernel invocation. Returns a list of
    {kernel_name, metrics: {metric: {value, unit}}} dicts — the shape
    handed to the model below."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if len(rows) < 3:
        return []
    header, units = rows[0], rows[1]
    kernels = []
    for row in rows[2:]:
        entry = {"kernel_name": row[0] if row else "", "metrics": {}}
        for col_idx in range(1, len(header)):
            if col_idx < len(row):
                entry["metrics"][header[col_idx]] = {"value": row[col_idx], "unit": units[col_idx]}
        kernels.append(entry)
    return kernels


def analyze_nsight_profile(kernels: list[dict]) -> list[dict]:
    """Sends parsed kernel metrics to Claude, gets back a ranked
    optimization list validated against SUGGESTION_SCHEMA. Never invoked
    in this repo -- see module docstring."""
    from anthropic import Anthropic

    client = Anthropic()
    prompt = (
        "Analyze these Nsight Compute per-kernel metrics (SM utilization, "
        "memory bandwidth, warp efficiency, and any other metrics present) "
        "and rank optimization opportunities from highest to lowest impact. "
        "Be specific: name the exact metric that motivates each suggestion "
        "and give concrete guidance (e.g. a tiling/occupancy change), not "
        "generic advice.\n\n" + json.dumps(kernels, indent=2)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": SUGGESTION_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["suggestions"]


if __name__ == "__main__":
    csv_text = sys.stdin.read() if not sys.stdin.isatty() else ""
    kernels = parse_ncu_metrics(csv_text)
    suggestions = analyze_nsight_profile(kernels)
    print(json.dumps(suggestions, indent=2))
