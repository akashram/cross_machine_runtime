# production_framework

**Status: code-complete AND locally run — `.venv` (`ray==2.49.2`), real
production-framework training run. DeepSpeed genuinely doesn't work on
this platform (verified, not assumed) — documented below, not hidden.**

## What this measures

PLAN.md Phase 19 step 6: one real run through a production training
framework — DeepSpeed attempted FIRST per PLAN.md's explicit instruction,
with any genuine Mac/CPU incompatibility "verified empirically and
documented honestly rather than assumed," falling back to Lightning/Ray
Train only if DeepSpeed's install or feature set turns out to be genuinely
gated.

## DeepSpeed: attempted, hit two real, independent walls

1. **`pip install deepspeed` succeeds**, but `import deepspeed` crashes
   with `ModuleNotFoundError: No module named 'distutils'` —
   `deepspeed/ops/op_builder/builder.py` imports the standard-library
   `distutils` module unconditionally at import time, and `distutils` was
   REMOVED from the Python standard library in 3.12 (this `.venv`'s
   Python version). Worked around by pinning `setuptools<72` (older
   `setuptools` versions vendor a compatible `distutils` shim).
2. **With that fixed, `import deepspeed` hits a SECOND, deeper wall**:
   `deepspeed/runtime/zero/muon/original_muon.py` decorates a function
   with DeepSpeed's own `@compiler.compile()` wrapper AT MODULE IMPORT
   TIME (not lazily, not only when actually used) — which calls
   `torch.compile()` internally and hits the EXACT SAME
   `RuntimeError: Dynamo is not supported on Python 3.12+` wall step 2
   (`torch_compile_bench`) already found and documented. This one is not
   workaroundable without patching DeepSpeed's own source: it's baked
   into an unconditional import-time decorator inside a third-party
   package, not something this repo's code controls.

**Verdict: DeepSpeed does not run on this platform/Python/torch
combination**, confirmed by two independent, real failures (not one
ambiguous one), not assumed from a compatibility table.

## Ray Train: the real, working fallback

`ray[train]` (an extras install on top of the already-installed `ray`,
not a new package) provides `TorchTrainer`, which wires up real
`DistributedDataParallel` across worker processes automatically. Trains
the SAME architecture as step 1's `transformer_torch.py` — deliberately
made a self-contained copy in this file rather than importing across the
driver/worker process boundary (see the second real bug below).

## Results (captured 2026-08-09, `ray==2.49.2`, this Mac)

```
Training finished iteration 1 at 2026-08-09 17:29:31. Total running time: 14s
╭───────────────────────────────╮
│ Training result               │
├───────────────────────────────┤
│ time_total_s           11.027 │
│ first_loss            2.96396 │
│ last_loss             0.02637 │
╰───────────────────────────────╯

  Ray Train TorchTrainer, 2 real DDP workers (Ray-managed, not manually wired like step 3):
  corpus: 'the quick fox jumps '
  Ray Train: loss 2.9640 -> 0.0264
  C++ (transformer_test.cpp, real captured):  loss 3.1891 -> 0.0171

PASS  Ray Train's final loss < 0.1 (C++ reached 0.0171), a real production-framework training run completed end to end
```

## Findings

- **A real, second bug, distinct from DeepSpeed's**: Ray Train workers run
  in SEPARATE actor processes, each with their own working directory. A
  driver-side `sys.path.insert(0, "../pytorch_transformer")` plus
  `from transformer_torch import ...` (the natural first thing to try,
  matching how `torch_compile_bench.py` and `ray_train_run.py`'s own
  first draft reused code from sibling directories) works fine in the
  driver process but fails inside every Ray worker with
  `ModuleNotFoundError: No module named 'transformer_torch'`, since the
  relative `sys.path` entry doesn't transfer across the actor-process
  boundary. Rather than fight Ray's `runtime_env`/`py_modules`
  module-shipping mechanics further, this file makes itself
  self-contained — the model architecture duplicated locally instead of
  imported — a deliberate, disclosed trade of a little code duplication
  for robustness.
- **Ray Train's real DDP wiring reproduces step 1's PyTorch numbers
  closely**: `2.9640 -> 0.0264` here vs. step 1's standalone PyTorch run's
  `2.9640 -> 0.0263` — same architecture, same corpus, same seed, so this
  is effectively the same training run, now going through Ray Train's
  real distributed orchestration (2 actual DDP worker processes) instead
  of a single process, and landing in the same place.
- **This step's real finding is about the ATTEMPT, not just the
  fallback**: PLAN.md explicitly asked for DeepSpeed to be tried first and
  any incompatibility verified empirically, not assumed — and the
  empirical result is two independent, real, well-documented reasons
  DeepSpeed doesn't run here (a Python 3.12 stdlib removal, and
  DeepSpeed's own unconditional `torch.compile` usage at import time
  hitting the same Dynamo/Python-3.12 wall step 2 found). That's a more
  useful, concrete result than either "DeepSpeed works" (false) or
  "DeepSpeed is skipped, assumed incompatible" (unverified) would have
  been.

## Hardware notes
CPU only. Ray Train's 2 workers are both local processes on this Mac —
no multi-machine Ray cluster involved or needed for this step's claim.
