#!/usr/bin/env bash
# provision_tpu_vm.sh — create a GCP TPU VM and install JAX with TPU support.
#
# PLAN.md Phase 8 step 1. Real gcloud commands, meant to be run from a
# workstation with the gcloud CLI authenticated against a project that has
# TPU quota (either paid, or an approved TRC — Tpu Research Cloud — grant).
# Unrun here: no GCP project/credentials/quota on this Mac.
#
# Usage: PROJECT=my-project ./provision_tpu_vm.sh

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=<gcp-project-id>}"
ZONE="${ZONE:-us-central2-b}"           # us-central2 has v4 capacity
TPU_NAME="${TPU_NAME:-cmr-tpu-v4-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v4-8}"   # cheapest multi-chip slice
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"

echo "== creating TPU VM: ${TPU_NAME} (${ACCELERATOR_TYPE}) in ${ZONE} =="
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --version="${RUNTIME_VERSION}"

echo "== installing JAX with TPU support on all workers =="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

echo "== copying repo and running the matmul validation script =="
gcloud compute tpus tpu-vm scp "$(dirname "$0")/validate_matmul.py" \
  "${TPU_NAME}:~/validate_matmul.py" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="python3 ~/validate_matmul.py"

cat <<EOF

== done ==
Teardown when finished (TPU VMs bill by the second while running):
  gcloud compute tpus tpu-vm delete ${TPU_NAME} --project=${PROJECT} --zone=${ZONE}
EOF
