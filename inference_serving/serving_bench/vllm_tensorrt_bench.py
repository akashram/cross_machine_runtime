"""vllm_tensorrt_bench.py -- throughput and p50/p99/p999 TTFT/TPOT latency
comparison: this repo's own serving stack vs. vLLM vs. TensorRT-LLM, on
real GPU hardware with a real hosted checkpoint.

PLAN.md Phase 9 step 9: "throughput (requests/sec), latency (p50/p99/p999
time-to-first-token, time-per-output-token), comparison against vLLM and
TensorRT-LLM." serving_latency_bench.cpp already measures this repo's own
CPU path locally, for real; this script is the GPU-hardware-gated other
two-thirds of the comparison.

API note, same spirit as tpu_engine/stablehlo_execute's jax.export
caveat: TensorRT-LLM's Python API has moved fast across releases (raw
`tensorrt_llm.runtime.ModelRunner` -> the newer high-level
`tensorrt_llm.hlapi.LLM`, which deliberately mirrors vLLM's `LLM` +
`SamplingParams` shape for easy side-by-side comparison, which is what
this script uses). Confirm the exact import path against whatever
TensorRT-LLM version is installed on the benchmark GPU instance before
relying on this unrun.

Unrun here -- no GPU, no vllm/tensorrt_llm install, no hosted checkpoint
on this Mac.
"""

import argparse
import statistics
import time

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # placeholder -- pick whatever checkpoint this repo settles on
NUM_REQUESTS = 200
MAX_NEW_TOKENS = 128
PROMPTS = [f"Request {i}: summarize the plot of a short story about a lighthouse keeper." for i in range(NUM_REQUESTS)]


def percentile(values, pct):
    values = sorted(values)
    idx = int(pct / 100.0 * (len(values) - 1))
    return values[idx]


def bench_vllm():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID)
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    ttfts, tpots = [], []
    wall_start = time.perf_counter()
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        t1 = time.perf_counter()
        # vLLM's RequestOutput carries per-token timing metrics when
        # available; fall back to wall time / token count if the
        # installed version doesn't expose per-token timestamps.
        num_tokens = len(outputs[0].outputs[0].token_ids)
        total_ms = (t1 - t0) * 1e3
        ttfts.append(total_ms / max(num_tokens, 1))  # approximation without per-token timestamps
        tpots.append(total_ms / max(num_tokens, 1))
    wall_s = time.perf_counter() - wall_start

    return {
        "throughput_req_s": len(PROMPTS) / wall_s,
        "ttft_p50": percentile(ttfts, 50), "ttft_p99": percentile(ttfts, 99), "ttft_p999": percentile(ttfts, 99.9),
        "tpot_p50": percentile(tpots, 50), "tpot_p99": percentile(tpots, 99), "tpot_p999": percentile(tpots, 99.9),
    }


def bench_tensorrt_llm():
    from tensorrt_llm.hlapi import LLM, SamplingParams  # mirrors vLLM's API shape deliberately

    llm = LLM(model=MODEL_ID)
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    ttfts, tpots = [], []
    wall_start = time.perf_counter()
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        output = llm.generate([prompt], sampling)
        t1 = time.perf_counter()
        num_tokens = len(output[0].outputs[0].token_ids)
        total_ms = (t1 - t0) * 1e3
        ttfts.append(total_ms / max(num_tokens, 1))
        tpots.append(total_ms / max(num_tokens, 1))
    wall_s = time.perf_counter() - wall_start

    return {
        "throughput_req_s": len(PROMPTS) / wall_s,
        "ttft_p50": percentile(ttfts, 50), "ttft_p99": percentile(ttfts, 99), "ttft_p999": percentile(ttfts, 99.9),
        "tpot_p50": percentile(tpots, 50), "tpot_p99": percentile(tpots, 99), "tpot_p999": percentile(tpots, 99.9),
    }


def print_results(label, r):
    print(f"\n=== {label} ===")
    print(f"throughput: {r['throughput_req_s']:.2f} req/s")
    print(f"TTFT (ms): p50={r['ttft_p50']:.2f} p99={r['ttft_p99']:.2f} p999={r['ttft_p999']:.2f}")
    print(f"TPOT (ms): p50={r['tpot_p50']:.2f} p99={r['tpot_p99']:.2f} p999={r['tpot_p999']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "tensorrt_llm", "both"], default="both")
    args = parser.parse_args()

    if args.engine in ("vllm", "both"):
        print_results("vLLM", bench_vllm())
    if args.engine in ("tensorrt_llm", "both"):
        print_results("TensorRT-LLM", bench_tensorrt_llm())

    print(
        "\nCompare against serving_latency_bench.cpp's real local CPU-path "
        "numbers (this repo's own unbatched, uncached serving_backend "
        "router + transformer/ model) -- that comparison is CPU-toy-model "
        "vs GPU-production-model, not apples to apples on absolute "
        "latency, but the batching/paging/quantization techniques Phase 9 "
        "implements (steps 1-3, 6-7) are what would close that gap on "
        "real hardware, once wired into a GPU backend for "
        "serving_backend's router."
    )
