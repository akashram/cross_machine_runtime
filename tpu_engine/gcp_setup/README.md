# gcp_setup

**Status: code-complete, hardware-gated — real gcloud/JAX code, matching
Phase 3's convention for GPU code that hasn't run yet. Unrun: no GCP
project, TPU quota, or TRC grant available from this Mac.**

## What this measures

PLAN.md Phase 8 step 1: provision a TPU VM, install JAX with TPU support,
and validate the setup with a simple matmul (device visibility, numerical
correctness vs. a CPU/numpy reference, and a rough TFLOPS number).

## Design

- `provision_tpu_vm.sh`: real `gcloud compute tpus tpu-vm` commands —
  create a `v4-8` slice (cheapest multi-chip entry point per PLAN.md),
  install `jax[tpu]` on every worker, copy over and run the validation
  script, and print the teardown command (TPU VMs bill by the second
  while running, so this is called out explicitly rather than left
  implicit).
- `validate_matmul.py`: three checks in order — `jax.devices()` /
  `jax.default_backend()` actually report `tpu`; a `jax.jit`-compiled
  matmul matches a `numpy` reference within tolerance; a bf16 matmul
  throughput number as a first sanity data point (not a tuned benchmark —
  that's step 2, `tpu_benchmarks`).
- No portable local subset: `jax.devices()` on this Mac reports a CPU
  device, so the one check this script exists to make (`backend == "tpu"`)
  would trivially fail here for the right reason, not a bug. Running it
  locally would validate nothing PLAN.md step 1 asks for.

## TRC application note

PLAN.md flags applying for Tpu Research Cloud (TRC) "early (before this
phase)" — it's a manual web form + approval wait, not something this repo
can automate. Apply before starting Phase 8's hardware validation pass so
the wait overlaps with other phases' hardware validation instead of
blocking Phase 8 specifically.

## Results
TODO: run on a GCP TPU VM (TRC grant or paid `v4-8`).

| Check | Result |
|-------|--------|
| backend == tpu | TODO |
| matmul max abs err vs numpy | TODO |
| bf16 4096x4096 TFLOPS | TODO |

## Hardware notes
- Required: GCP project with TPU quota (TRC free grant, or paid `v4-8`
  minimum), `gcloud` CLI authenticated locally.
- `us-central2` used as the default zone — has v4 capacity per GCP docs;
  override with `ZONE=` if a TRC grant lands you in a different region.
