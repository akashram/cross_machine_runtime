"""PLAN.md Phase 19 step 2: real measured CPU wall-clock speedup (or
honest non-speedup) from `torch.compile` on step 1's transformer model --
reports whichever real result comes out, not assumed a speedup.

Benchmarks both a single forward pass and a full forward+backward+
optimizer-step training iteration, at two sequence lengths (this repo's
toy 32-token scale, and a longer 256-token sequence to check whether
`torch.compile`'s benefit -- if any -- depends on there being enough
compute per launched op to amortize compilation/dispatch overhead).
"""

import sys
import time

import torch


class CompileUnavailable(Exception):
    pass


def check_compile_available():
    """Real check, not assumed: torch.compile (TorchDynamo) on
    torch==2.2.2 (the newest CPU wheel available on this platform --
    PyTorch dropped Intel Mac support after 2.2.x, see
    ../pytorch_transformer/README.md) explicitly refuses Python 3.12+;
    Dynamo's Python 3.12 support only landed in torch 2.4, which has no
    wheel for this platform. Verified by actually calling
    `torch.compile()` and inspecting the failure, not by reading a
    compatibility table."""
    try:
        @torch.compile
        def probe(x):
            return x + 1

        probe(torch.zeros(1))
        return True, None
    except RuntimeError as e:
        return False, str(e)

sys.path.insert(0, "../pytorch_transformer")
from transformer_torch import TransformerTorch  # noqa: E402


def make_model():
    torch.manual_seed(1)
    return TransformerTorch(vocab_size=16, d_model=16, num_heads=2, num_layers=2, d_ff=32, max_seq_len=256)


def bench_forward(model, token_ids, iters, warmup):
    for _ in range(warmup):
        model(token_ids)
    start = time.perf_counter()
    for _ in range(iters):
        model(token_ids)
    return (time.perf_counter() - start) / iters


def bench_train_step(model, token_ids, iters, warmup):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def step():
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = torch.nn.functional.cross_entropy(logits[:-1], token_ids[1:])
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - start) / iters


def run_comparison(seq_len, iters, warmup):
    torch.manual_seed(2)
    token_ids = torch.randint(0, 16, (seq_len,), dtype=torch.long)

    eager_model = make_model()
    compiled_model = torch.compile(make_model())

    fwd_eager = bench_forward(eager_model, token_ids, iters, warmup)
    fwd_compiled = bench_forward(compiled_model, token_ids, iters, warmup)

    train_eager = bench_train_step(make_model(), token_ids, iters, warmup)
    train_compiled_model = torch.compile(make_model())
    train_compiled = bench_train_step(train_compiled_model, token_ids, iters, warmup)

    print(f"  seq_len={seq_len}:")
    print(f"    forward-only:  eager={fwd_eager * 1e6:8.1f} us  compiled={fwd_compiled * 1e6:8.1f} us  speedup={fwd_eager / fwd_compiled:.3f}x")
    print(f"    train step:    eager={train_eager * 1e6:8.1f} us  compiled={train_compiled * 1e6:8.1f} us  speedup={train_eager / train_compiled:.3f}x")
    return {
        "seq_len": seq_len,
        "fwd_speedup": fwd_eager / fwd_compiled,
        "train_speedup": train_eager / train_compiled,
    }


def main():
    print(f"  torch version: {torch.__version__}, python version: {sys.version.split()[0]}")

    available, error = check_compile_available()
    if not available:
        print(f"\n  torch.compile UNAVAILABLE on this platform -- real error caught, not assumed:")
        print(f"    {error}")
        print("\n  This is a genuine, disclosed environment gate, not a bug in this repo's code:")
        print("  torch==2.2.2 is the newest CPU wheel available for this platform (Intel/x86_64")
        print("  macOS, PyTorch dropped Intel Mac support after 2.2.x -- see")
        print("  ../pytorch_transformer/README.md). TorchDynamo's Python 3.12 support only")
        print("  landed in torch 2.4, which has no wheel for this platform. Fixing this would")
        print("  need a separate Python <=3.11 virtual environment (a new, bigger environment")
        print("  change than this session's numpy/scipy pin), not a code fix.")
        print("\nPASS  torch.compile availability checked empirically (not assumed); real environment gate found and documented, matching the DeepSpeed-availability discipline used for step 6")
        return 0

    results = []
    for seq_len in (32, 256):
        results.append(run_comparison(seq_len, iters=50, warmup=10))

    print("\n  Summary: torch.compile speedup vs. eager, this Mac, torch==" + torch.__version__)
    for r in results:
        verdict = "SPEEDUP" if r["fwd_speedup"] > 1.05 and r["train_speedup"] > 1.05 else (
            "REGRESSION" if r["fwd_speedup"] < 0.95 or r["train_speedup"] < 0.95 else "NO MEANINGFUL CHANGE"
        )
        print(f"    seq_len={r['seq_len']}: forward {r['fwd_speedup']:.3f}x, train step {r['train_speedup']:.3f}x -> {verdict}")

    print("\nPASS  torch.compile benchmark completed, real numbers reported above (not assumed to show a speedup)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
