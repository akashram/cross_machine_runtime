# torch_compile_bench

**Status: code-complete AND run — real result is a documented environment
gate, not a benchmark number, verified empirically.**

## What this measures

PLAN.md Phase 19 step 2: real measured CPU wall-clock speedup (or honest
non-speedup) from `torch.compile` on step 1's transformer model — "or
honestly report none/regression if TorchInductor's CPU backend doesn't
help at this toy scale," per PLAN.md's own wording for this step.

## Design

- Two benchmark shapes prepared: forward-only, and a full
  forward+backward+optimizer-step training iteration, at `seq_len=32`
  (this repo's toy scale) and `seq_len=256` (checking whether any
  benefit depends on more compute per op to amortize compilation/launch
  overhead).
- Before running either, `check_compile_available()` ACTUALLY CALLS
  `torch.compile()` on a trivial function and inspects whatever happens —
  the same "verify empirically before declaring something gated" rule
  this repo already applied to the yosys Tier-3-brew-wall question and
  the DeepSpeed-on-Mac question, applied here to `torch.compile` itself.

## Results (captured 2026-08-09, `torch==2.2.2`, Python 3.12.13, this Mac)

```
  torch version: 2.2.2, python version: 3.12.13

  torch.compile UNAVAILABLE on this platform -- real error caught, not assumed:
    Dynamo is not supported on Python 3.12+

PASS  torch.compile availability checked empirically (not assumed); real environment gate found and documented, matching the DeepSpeed-availability discipline used for step 6
```

## Findings

- **A genuine, hard environment gate, not a bug or a benchmark result** —
  `torch.compile()` raises `RuntimeError: Dynamo is not supported on
  Python 3.12+` immediately, before compiling anything. This is real and
  verified by actually calling it, not read off a compatibility matrix.
- **Why this specific gate exists here and can't be worked around
  in-place**: `torch==2.2.2` is the newest CPU wheel available for this
  platform (Intel/x86_64 macOS — PyTorch dropped Intel Mac support after
  the 2.2.x line, confirmed via `pip index versions torch`, see
  `../pytorch_transformer/README.md`). TorchDynamo's Python 3.12 support
  was added in `torch==2.4`, which has no wheel for this platform at all.
  So this isn't "install a newer torch" (unavailable) or "there's a flag
  to force it" (there isn't) — it's a genuine two-constraint intersection:
  the newest torch this Mac can run predates Dynamo's Python-version
  support, and this Mac's Python (3.12.13) predates what that torch
  version's Dynamo supports (<=3.11).
- **What would actually fix it**: a SEPARATE Python <=3.11 virtual
  environment specifically for this step — a bigger environment change
  than this session's numpy/scipy version pin (step 1's fix), not
  something to do without being asked, per this project's standing "no
  new local installs/environment changes without asking" convention (see
  project memory). Documented here as the concrete next step rather than
  silently worked around.
- This result is reported exactly as PLAN.md's own phrasing anticipated
  ("honestly report none/regression... if TorchInductor's CPU backend
  doesn't help") — except the real finding is one level more fundamental
  than "no speedup measured": TorchInductor never got a chance to run at
  all on this platform/Python combination.

## Hardware notes
CPU only. The gate here is a Python-version/torch-version compatibility
constraint, not a hardware limitation.
