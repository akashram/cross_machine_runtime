# Step 24: Chaos engineering harness — shell scripts only (inject.sh,
# node_kill.sh, gpu_oom.sh, chaos_run.sh), no compiled target. Requires
# Linux + tc (iproute2) for network faults; nvidia-smi/CUDA for GPU OOM;
# a live multi-node cluster for chaos_run.sh's orchestration.
