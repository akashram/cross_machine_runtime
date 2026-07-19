# Chaos Engineering Harness

**Status: scripts complete, not yet run — requires Linux with tc (traffic control) and a live multi-node cluster.**

## What this measures
Injects: network latency/loss/reordering via `tc netem`, node kills, GPU OOM.
Measures recovery time and correctness after each fault injection.

## Design
Four scripts, each independently invocable and composed by the last:
- `inject.sh` — `tc netem` faults: latency, loss, reorder (needs a base
  delay to have anything to reorder *relative to* — netem's own
  requirement, not a design choice here), corrupt, and partition (100%
  loss, kept as a distinct case name for readability from
  `chaos_run.sh` even though it's mechanically identical to `loss 100%`).
- `node_kill.sh` — SIGKILL, not SIGTERM: every component in this project
  already handles graceful shutdown correctly (raft's `stop()`,
  chandy_lamport's, etc.) — the interesting failure mode for a chaos
  harness is the *ungraceful* one, no chance to flush state or notify
  peers.
- `gpu_oom.sh` — documents the intended invocation (a `gpu_oom_inject`
  CUDA binary from `gpu_engine/memory/`, allocating until failure) rather
  than duplicating a CUDA allocation loop here; that binary doesn't exist
  yet because `gpu_engine/` hasn't run on real GPU hardware. Honest about
  what's missing rather than faking a result.
- `chaos_run.sh` — orchestrates one full cycle: confirm healthy, inject,
  poll a caller-supplied health-check command every 200ms up to 30s,
  restore, report recovery time. This is explicitly the next step once a
  live multi-node cluster exists — this project's local tests
  (`raft_test`, `chandy_lamport_test`) deliberately simulate multi-node
  with threads in one process so they're runnable without one; this
  script is for the real thing.

## Results
TODO: run on multi-node setup.

| Fault | Recovery time (s) | Data loss? | Correctness after recovery |
|-------|-----------------|------------|---------------------------|
| 100ms latency injection | N/A (soft) | No | TODO |
| 10% packet loss | TODO | No | TODO |
| Leader node kill | TODO | No | TODO |
| Network partition | TODO | No | TODO |

## Hardware notes
- Required: Linux with tc (iproute2), multi-node cluster
