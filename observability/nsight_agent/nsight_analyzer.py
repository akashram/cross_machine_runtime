#!/usr/bin/env python3
"""
nsight_analyzer.py — AI agent that parses Nsight Compute profiles and ranks
optimization suggestions using the Claude API.

TODO: implement using Claude API (claude-opus-4-8 or claude-sonnet-4-6)

Workflow:
1. Parse ncu CSV/JSON output (from nsight_ci/parse_ncu.py)
2. Send metrics to Claude with a structured prompt
3. Claude returns ranked optimization suggestions with specific guidance
4. Output: JSON list of {priority, kernel, metric, suggestion, expected_gain}
"""
import json
import sys

def analyze_nsight_profile(metrics_json: dict) -> list[dict]:
    """
    Call Claude API to analyze Nsight metrics and generate suggestions.
    TODO: implement on GPU instance with Anthropic API key set.
    """
    # from anthropic import Anthropic
    # client = Anthropic()
    # prompt = f"Analyze these Nsight Compute metrics and rank optimization opportunities:\n{json.dumps(metrics_json, indent=2)}"
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=2048,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return json.loads(response.content[0].text)
    print("nsight_analyzer: STUB — implement with Anthropic API key on GPU instance")
    return []

if __name__ == "__main__":
    metrics = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    suggestions = analyze_nsight_profile(metrics)
    print(json.dumps(suggestions, indent=2))
