# Containerized & Orchestrated Deployment — Design

See `README.md` for the per-step status table. This document covers the
motivation, the CMakeLists.txt tracing behind the Dockerfile's dependency
set, the two genuine gaps this pass found (and disclosed rather than
papered over), and points to the device-plugin/scheduler comparisons.

## 1. Motivation: a real, concrete CI-drift finding, not a hypothetical

`.github/workflows/ci.yml` has been `workflow_dispatch`-only since
2026-07-29 because it fails fast (~35-46s) on every push, and its own
comment says why: it predates almost every toolchain this repo now
depends on (gRPC, FlatBuffers, libfabric, BCC/libbpf, MLIR) and was never
updated as those got added. This phase's Dockerfile (step 1) is the
actual reconciliation exercise: "what does this tree need to build,
really" — traced from the CMakeLists.txt files themselves, not
reconstructed from memory of what got added when.

## 2. What tracing the CMakeLists.txt tree actually found

The result was more optimistic than ci.yml's own framing suggests. Every
optional dependency in the tree (Protobuf, gRPC, FlatBuffers, libfabric,
libbpf) is behind `find_package(... QUIET)` / `find_library(...)` +
a graceful `return()` if missing — confirmed by grepping the whole tree
for `REQUIRED`: the only unguarded ones are `find_package(Threads
REQUIRED)` (ubiquitous on any Linux base image) plus CUDAToolkit/MLIR/
Stablehlo, each already gated one level up by the root
`CMakeLists.txt`'s `check_language(CUDA)` / `if(DEFINED MLIR_DIR)`, or by
the subdirectory's own `if(DEFINED MLIR_DIR AND DEFINED STABLEHLO_DIR)`
(`tpu_engine/stablehlo_lower`) / `if(NOT CMAKE_CUDA_COMPILER) ... return()`
(`distributed_training/gpudirect_storage`). `vcpkg.json` declares zero
dependencies. Net finding: **this entire tree already configures and
builds on a bare Linux box with nothing but a C++23 compiler, CMake
>= 3.25, and Ninja** — every one of the "new toolchain" dependencies
ci.yml's comment blames just silently skips its handful of targets
without them.

That raised the real question: if the tree already tolerates missing
toolchains gracefully, why would ci.yml fail at configure time at all?
Grepping further: `CMakeLists.txt` pins `cmake_minimum_required(VERSION
3.25)` (matching `CMakePresets.json`), but ci.yml's `apt-get install
ninja-build clang` step never installs or pins `cmake` itself — it
trusts whatever cmake ships preinstalled on the `ubuntu-latest` runner
image. Ubuntu 22.04's own apt `cmake` package is 3.22.x, below the
project's minimum; if `ubuntu-latest` was resolving to that Ubuntu
version (or any apt snapshot below 3.25) at the time ci.yml started
failing, `cmake --preset debug` would fail immediately with a version
error — a fast, configure-time failure with zero relation to any of the
gRPC/FlatBuffers/libfabric/libbpf/MLIR toolchains actually named in
ci.yml's comment. This fork had no `gh` CLI available to check the real
Actions run logs, so this is presented as a well-evidenced hypothesis,
not a confirmed diagnosis — but `Dockerfile` sidesteps the question
regardless by installing a modern `cmake` explicitly via `pip3 install
"cmake>=3.28"` rather than trusting any base image's apt package.

## 3. Dockerfile package choices, and why installing the "optional" ones anyway

`Dockerfile`'s first `apt-get` layer (clang, ninja-build, a pip-installed
modern cmake) is the minimum for the already-portable baseline above. Its
second layer (libprotobuf-dev/protobuf-compiler, libgrpc++-dev, libflat
buffers-dev/flatbuffers-compiler, libfabric-dev, libbpf-dev) is not
required by that baseline at all — every one of those targets already
degrades gracefully without them. They're installed anyway because a
real Linux container is exactly the environment that CAN build those
five additional targets for real
(`networking/grpc_control`, `networking/flatbuffers_data`,
`networking/af_xdp`, `networking/userspace_net`,
`networking/rdma_onesided`) where this Mac never could — leaving them
out would silently under-sell what "the portable build, containerized"
should mean.

## 4. Two genuine gaps this pass found, disclosed rather than hidden

Both surfaced while writing the Kubernetes manifests (steps 4/5), from
actually reading the relevant source rather than assuming a server/worker
entrypoint existed:

- **No long-running serving daemon** (`k8s/serving/deployment.yaml`):
  `inference_serving/serving_backend/` has a real `ServingRouter` with
  correct dispatch/fallback logic (see its own README), but the only
  binary that exists is `serving_router_test` — a unit test that asserts
  and exits. A Kubernetes `Deployment` fundamentally assumes a
  long-running process (readiness probes, restart-on-crash, steady
  traffic); there is nothing in this repo yet that fits that shape. The
  manifest is written for the entrypoint a real `serving_daemon` (HTTP/
  gRPC wrapper around `ServingRouter::route()`) would need, explicitly
  flagged as not-yet-built.
- **No rank-per-process training entrypoint**
  (`k8s/training/statefulset.yaml`): grepping
  `networking/ring_allreduce`, `networking/collectives`, and
  `distributed_training/full_training_loop` for `RANK`/`WORLD_SIZE`/
  `getenv` turns up nothing — every multi-rank step in this repo
  simulates all N ranks as THREADS inside one OS process talking over
  loopback TCP, proven correct that way, but with no existing binary
  that behaves as a single rank across a real process/pod boundary. The
  gang-scheduling MECHANISM (`podManagementPolicy: Parallel` +
  `StatefulSet` ordinal-derived hostnames via a headless Service) is
  real and correctly solves the synchronized-start problem PLAN.md asks
  about; the `training_worker` binary it launches is the shape such an
  entrypoint would need, not something that exists today.

Both gaps are scoped, real follow-on work (write a thin process-boundary
wrapper around the already-correct `Autograd`/`RingAllreduce`/
`ServingRouter` logic) — not attempted as part of this pass, since doing
so honestly would mean modifying `distributed_training/` and
`inference_serving/` source, a larger change than "add containers/
orchestration artifacts for the existing tree."

## 5. Judgment call: StatefulSet, not Kubeflow `PyTorchJob`

`k8s/training/statefulset.yaml` documents this inline, summarized here:
`PyTorchJob` is a CRD requiring the Kubeflow training-operator be
installed — a real, separate cluster dependency this repo can't assume,
unlike `StatefulSet` (built into every Kubernetes distribution). It's
also a poor conceptual fit regardless of availability: this repo's
training stack is a from-scratch autograd + collectives implementation
(`distributed_training/autograd/`), not a PyTorch program, so
`PyTorchJob`'s specific value (torch.distributed env-var injection,
torchrun-aware restart semantics) doesn't apply. A generic
gang-start primitive is the honest fit.

## 6. Device plugin gap analysis and scheduler comparison

Full write-ups, not summarized further here: `k8s/device_plugin_gap_
analysis.md` (step 6 — NVIDIA's device-plugin model fits GPUs, is a poor
structural fit for FPGA's partial-reconfiguration state, and doesn't
apply to TPU at all, which uses dedicated node pools instead) and
`k8s/scheduler_comparison.md` (step 7 — `topo_scheduler` is justified,
narrowly, for fine-grained measured-bandwidth placement Kubernetes'
label-hierarchy model can't express; `multitenancy` is NOT justified as
a cluster-level scheduler, since Kueue/Volcano already do weighted
fair-sharing with quota borrowing better, and is only defensible at the
in-process request-fairness granularity Kubernetes doesn't reach at
all).

## 7. What's NOT done in this pass

- Step 2 (cgroup interaction measured) — see `README.md`'s explanation of
  why this is a bare deferral rather than a written-but-unrun artifact.
- Actually running any of steps 1/3/4/5 — no Docker, GPU, or Kubernetes
  cluster available locally.
- The two entrypoint gaps in §4 — real follow-on work, not attempted
  here.
