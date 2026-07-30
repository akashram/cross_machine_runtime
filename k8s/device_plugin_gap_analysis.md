# Device plugin gap analysis — NVIDIA vs. Xilinx FPGA vs. TPU

Phase 16 step 6. Pure written analysis, no cluster or hardware needed to
produce it — grounded in each vendor's actual, publicly documented device
exposure model, not a generic "NVIDIA good, Xilinx less standardized"
hand-wave.

## The Kubernetes device plugin model, briefly

The device plugin API (`k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1`) lets
a DaemonSet advertise a countable, interchangeable extended resource (like
`nvidia.com/gpu: 4`) to the kubelet, which then hands the scheduler a
simple bin-packing problem: find a node with >= N free units of that
resource. The model's built-in assumption: every unit of the resource is
functionally identical and stateless between pod assignments (a GPU
freed by one pod is exactly as good as any other free GPU for the next
pod). That assumption is exactly right for one of this repo's three
non-CPU/GPU-training backends and exactly wrong for another.

## NVIDIA: the model fits, and the tooling is mature

`NVIDIA/k8s-device-plugin` implements the device plugin API directly,
advertising `nvidia.com/gpu` (and, with MIG enabled on A100/H100-class
cards, finer-grained MIG-instance resources like
`nvidia.com/mig-1g.5gb`). It composes with `nvidia-container-toolkit`
(this repo's own `docker/gpu/daemon.json`, step 3) for the actual device
node passthrough at container-start time. NVIDIA packages the whole
lifecycle — driver install, device plugin, DCGM-based monitoring,
MIG-partition management, time-slicing config — as the "GPU Operator," a
single Helm-installable Operator most managed Kubernetes offerings
(GKE, EKS, AKS) document directly. This is the device-plugin model
working as designed: a GPU genuinely is a countable, stateless-between-
assignments unit once MIG/time-slicing config is fixed.

## Xilinx/AMD FPGA: the model is a poor architectural fit, not just less mature tooling

Xilinx has published reference FPGA device plugin implementations (e.g.
community Xilinx/AMD FPGA device-plugin projects advertising something
like `xilinx.com/fpga-xilinx_u250_gen3x16-0`), but the gap here is
deeper than "NVIDIA's plugin has more GitHub stars." The device plugin
model's core assumption — a unit of the resource is interchangeable and
stateless between assignments — is false for exactly the FPGA capability
this repo built in Phase 7 step 22
(`fpga_engine/partial_reconfig/`, `dfx_pblock.tcl`): partial
reconfiguration means "the FPGA" isn't one fixed device, it's a static
region plus one or more reconfigurable pblocks whose loaded bitstream
(RM_A vs. RM_B in this repo's own DFX example) is real, persistent
runtime state. A device plugin advertising "1 FPGA available" the way it
advertises "1 GPU available" throws away exactly the information a
scheduler would need to answer "does this FPGA currently have the
bitstream this pod needs loaded, or would scheduling here cost a
`pr_host_driver.cpp`-measured multi-millisecond hot-swap first" (see
`fpga_engine/partial_reconfig/README.md`'s own 6.25ms modeled swap
latency). No public FPGA device plugin models per-region bitstream state
as a schedulable dimension — they all reduce to the same "N countable
identical units" abstraction NVIDIA's plugin uses, which is a correct
fit for a GPU and a lossy approximation for a partially-reconfigurable
FPGA. This is the honest, specific finding: the maturity gap is real (far
fewer production deployments, no equivalent of GPU Operator's bundled
lifecycle management), but even a hypothetically NVIDIA-quality Xilinx
plugin couldn't fully close it without a different resource-modeling
primitive than the device plugin API provides today.

## TPU: doesn't use the device plugin model at all

GKE does not expose TPUs as a device-plugin extended resource the way it
does GPUs. Instead, TPUs are provisioned as dedicated node pools with
TPU-specific machine types, and pods request them via nodeSelector/
node-affinity labels (`cloud.google.com/gke-tpu-accelerator`,
`cloud.google.com/gke-tpu-topology`) rather than a resource-count
request. The reason is structural, not a tooling gap: a single TPU
"slice" (e.g. this project's own `tpu_engine/` targets a v4-8 or larger)
can span multiple physical host machines that must be scheduled and
started together as one unit — architecturally the same synchronized-
multi-pod-start problem this repo's own Phase 16 step 5
(`k8s/training/statefulset.yaml`) had to solve by hand for
`distributed_training/`'s ranks, not a simple "give me N of resource X"
request a per-pod device plugin API can express. Google's own tooling
for this (JobSet-based multi-host TPU provisioning, or the older
node-pool + Job-indexed-completion pattern) is closer in spirit to step
5's StatefulSet-plus-headless-Service gang-start than to
`NVIDIA/k8s-device-plugin`'s per-pod resource counting.

## Summary table

| Backend | Exposure model | Maturity | Structural fit |
|---|---|---|---|
| GPU (NVIDIA) | Device plugin (`nvidia.com/gpu`), MIG-aware | Mature; GPU Operator bundles full lifecycle | Good — GPU units genuinely are interchangeable/stateless between assignments |
| FPGA (Xilinx) | Device plugin (community-maintained), coarse resource count | Immature; no GPU-Operator-equivalent bundled lifecycle | Poor — doesn't model per-region bitstream/PR state (`fpga_engine/partial_reconfig`'s real capability), which the device-plugin abstraction has no room for regardless of tooling polish |
| TPU (GCP) | Dedicated node pools + topology labels, not device plugin | Mature within GCP's own tooling, but GKE-specific | Different in kind — multi-host slice scheduling is a gang-scheduling problem (see `k8s/training/`, step 5), not a per-pod resource-count problem |
