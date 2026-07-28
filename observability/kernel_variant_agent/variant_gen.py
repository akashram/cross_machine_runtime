#!/usr/bin/env python3
"""variant_gen.py — given a kernel specification (op type, input shapes,
dtype), asks Claude to generate N CUDA kernel variants with different
tiling/vectorization parameters, compiles and benchmarks each with `nvcc`
+ CUDA events, and promotes the fastest.

PLAN.md Phase 10 step 8: "generates N variants with different
tiling/vectorization parameters using Claude API, compiles and benchmarks
each, promotes the winner. Documents the search process."

Real client code (current Anthropic Python SDK, structured outputs via
`output_config.format` so each variant's tiling parameters come back
typed, not parsed out of free-text). Never invoked from this repo -- see
README.md, same "real code, no actual API calls" treatment as step 7
(nsight_agent) and step 9 (llm_autotune). The compile/benchmark half is
additionally GPU-toolchain-gated (needs `nvcc` + a CUDA device) regardless
of the API question.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

MODEL = "claude-opus-4-8"  # see SKILL.md: default unless the user names another model

VARIANT_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tile_m": {"type": "integer"},
                    "tile_n": {"type": "integer"},
                    "tile_k": {"type": "integer"},
                    "unroll_factor": {"type": "integer"},
                    "block_dim_x": {"type": "integer"},
                    "block_dim_y": {"type": "integer"},
                    "rationale": {"type": "string", "description": "why this parameter combination is worth trying"},
                    "cuda_source": {"type": "string", "description": "complete, compilable .cu kernel + launch wrapper"},
                },
                "required": ["name", "tile_m", "tile_n", "tile_k", "unroll_factor", "block_dim_x", "block_dim_y",
                             "rationale", "cuda_source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["variants"],
    "additionalProperties": False,
}


def generate_variants(kernel_spec: dict, n_variants: int = 8) -> list[dict]:
    """Asks Claude for n_variants distinct tiling/vectorization variants
    of the kernel described by kernel_spec. Never invoked in this repo."""
    from anthropic import Anthropic

    client = Anthropic()
    prompt = (
        f"Generate {n_variants} distinct CUDA kernel variants for this operation:\n"
        f"{json.dumps(kernel_spec, indent=2)}\n\n"
        "Vary tile sizes, unroll factors, and block dimensions across the variants -- "
        "cover a real spread (e.g. small-tile/high-occupancy through large-tile/high-"
        "arithmetic-intensity), not N cosmetic tweaks of the same design. Each "
        "cuda_source must be a complete, compilable .cu file: the kernel plus a "
        "launch wrapper function the benchmark harness below can call."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": VARIANT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["variants"]


def compile_and_benchmark(variant: dict, work_dir: Path, iters: int = 100) -> dict:
    """Compiles one variant with nvcc and benchmarks it via CUDA events.
    Real subprocess-based compile+run, GPU-toolchain-gated (needs nvcc +
    a CUDA device) independent of the API question above."""
    src_path = work_dir / f"{variant['name']}.cu"
    src_path.write_text(variant["cuda_source"])
    bin_path = work_dir / variant["name"]

    compile_result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", str(src_path), "-o", str(bin_path)],
        capture_output=True, text=True,
    )
    if compile_result.returncode != 0:
        return {"name": variant["name"], "compiled": False, "error": compile_result.stderr, "latency_us": None}

    # The compiled binary is expected to print "LATENCY_US: <float>" after
    # running `iters` warmed-up launches timed with cudaEvent -- the
    # cuda_source Claude generates is prompted (implicitly, via the
    # kernel_spec + expected harness contract) to include that main().
    run_result = subprocess.run([str(bin_path), str(iters)], capture_output=True, text=True, timeout=60)
    match = re.search(r"LATENCY_US:\s*([\d.]+)", run_result.stdout)
    if not match:
        return {"name": variant["name"], "compiled": True, "error": "no LATENCY_US in output", "latency_us": None}
    return {"name": variant["name"], "compiled": True, "error": None, "latency_us": float(match.group(1))}


def promote_winner(variants: list[dict], results: list[dict]) -> dict | None:
    """Picks the fastest successfully-compiled-and-benchmarked variant,
    returns it paired with a documented search summary."""
    ok = [r for r in results if r["compiled"] and r["latency_us"] is not None]
    if not ok:
        return None
    winner_result = min(ok, key=lambda r: r["latency_us"])
    winner_variant = next(v for v in variants if v["name"] == winner_result["name"])
    return {"variant": winner_variant, "result": winner_result, "all_results": results}


def run_search(kernel_spec: dict, n_variants: int = 8) -> dict | None:
    variants = generate_variants(kernel_spec, n_variants)
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        results = [compile_and_benchmark(v, work_dir) for v in variants]
    return promote_winner(variants, results)


if __name__ == "__main__":
    spec = {"op": "gemm", "M": 4096, "N": 4096, "K": 4096, "dtype": "fp16"}
    outcome = run_search(spec)
    if outcome is None:
        print("no variant compiled and benchmarked successfully")
    else:
        print(f"winner: {outcome['variant']['name']} @ {outcome['result']['latency_us']:.2f}us")
        print(f"rationale: {outcome['variant']['rationale']}")
        for r in outcome["all_results"]:
            status = f"FAIL: {r['error']}" if r["error"] else f"{r['latency_us']:.2f}us"
            print(f"  {r['name']}: {status}")
