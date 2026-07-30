# Phase 16: Containerized & Orchestrated Deployment

**Status: 6 of 7 steps code-complete this pass (steps 1, 3, 4, 5, 6, 7);
step 2 explicitly deferred (needs Docker, not installed locally — see
below). Every artifact is real and correct but UNRUN: no Docker, no
kubectl, no GPU, and no Kubernetes cluster exist on the dev Mac this was
written on.** See `DESIGN.md` for the full rationale, the two genuine
gaps this pass found and disclosed (no per-process rank-launching
training entrypoint; no long-running serving daemon binary), and the
device-plugin / custom-scheduler comparisons in full.

Artifacts live at their conventional real-world locations rather than
nested under this directory, since Docker/Kubernetes tooling expects
specific paths (`Dockerfile` at repo root for build context; `k8s/` for
manifests) — this directory holds the phase-level write-up only, playing
the role individual per-step `README.md` files play in every other phase
(this phase doesn't decompose into one directory per step the way, say,
`fpga_engine/`'s steps do).

## Per-step status

| Step | What | Where | Status |
|---|---|---|---|
| 1. Dockerfile for the portable build | Multi-stage image (builder + runtime) for the CPU-portable tree subset | `Dockerfile` (repo root) | Real, complete, **unrun** — no Docker locally |
| 2. cgroup interaction measured | Run `cpu_engine`'s affinity/hugepage/NUMA code inside vs. outside a container, measure what breaks | — | **Deferred entirely** — this step's whole point is a *measured* finding from actually running containers; without Docker there is nothing honest to write beyond this line. Not attempted as a fake "test plan." |
| 3. GPU passthrough config | `nvidia-container-toolkit` daemon config + CUDA-base Dockerfile variant | `docker/gpu/` | Real, complete, **unrun** — hardware- AND tool-gated (no GPU, no Docker) |
| 4. K8s manifests: inference serving | Deployment + Service + latency-driven HPA for `ServingRouter` | `k8s/serving/` | Real, complete, **unrun** — no kubectl/cluster; also surfaces a real gap (no serving daemon binary exists yet, only a unit test) |
| 5. K8s manifests: gang-scheduled training | StatefulSet (`podManagementPolicy: Parallel`) + headless Service for `distributed_training/` | `k8s/training/` | Real, complete, **unrun**; surfaces a deeper gap (no rank-per-process training entrypoint exists — every multi-rank step in this repo simulates ranks as threads in one process) |
| 6. Device plugin gap analysis | NVIDIA vs. Xilinx FPGA vs. TPU device exposure models, compared specifically | `k8s/device_plugin_gap_analysis.md` | Written analysis, no hardware/tooling needed — done |
| 7. Custom schedulers vs. Kubernetes | `topo_scheduler`/`multitenancy` vs. Kubernetes-native + Kueue/Volcano, with a real, differentiated verdict per component | `k8s/scheduler_comparison.md` | Written analysis, no hardware/tooling needed — done |

## Why step 2 is a bare deferral, not a written-but-unrun artifact

Every other unrun artifact in this repo (this phase's steps 1/3/4/5
included) is a REAL, complete file whose correctness doesn't depend on
having actually executed it — a Dockerfile's package list, a Kubernetes
manifest's schema, a CUDA kernel's logic can all be checked by a careful
human/reviewer without running them, the same way `fpga_engine/vitis_ai`'s
`mlp_model.py` is real Python that just hasn't been run. Step 2 is
different in kind: its ENTIRE deliverable, per PLAN.md, is "document
concretely what breaks or needs explicit configuration" when running
inside a container — a claim that is either a real measurement or
worthless speculation, with no real middle ground the way a Dockerfile's
correctness has one. Writing a plausible-sounding "expected findings"
document here would look identical to a real one while being
substantively fake. Deferred cleanly instead, same treatment
`tpu_engine`'s JAX-dependent steps get for the analogous reason (see
`CLAUDE.md`'s Phase 8 status).

## What running this would need, concretely

- Step 1/2: Docker (or Colima/Podman) installed locally, then
  `docker build -t cross_machine_runtime:latest .` and
  `docker run --rm cross_machine_runtime:latest` at repo root.
- Step 3: the above, plus a real NVIDIA GPU + driver + the
  `nvidia-container-toolkit` package (`docker/gpu/daemon.json` documents
  the exact runtime registration), then
  `docker compose -f docker/gpu/docker-compose.gpu.yml up --build`.
- Steps 4/5: a real Kubernetes cluster (even `kind`/`minikube` locally)
  plus `kubectl apply -f k8s/serving/` / `k8s/training/`, and — per the
  gaps disclosed above and in `DESIGN.md` — a real `serving_daemon` and
  `training_worker` binary that don't exist in this repo yet.
