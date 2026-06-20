# Chaos Engineering Harness

**Status: STUB — requires Linux with tc (traffic control) and multi-node setup.**

## What this measures
Injects: network latency/loss/reordering via `tc netem`, node kills, GPU OOM.
Measures recovery time and correctness after each fault injection.

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
