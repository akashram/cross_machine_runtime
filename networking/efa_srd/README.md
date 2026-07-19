# EFA SRD vs RC Transport

**Status: code-complete AND locally run — pure decision logic, no
libfabric/EFA dependency.**

## What this measures
When EFA's SRD (Scalable Reliable Datagram) transport's scalable
one-QP-to-many-peers model wins over RC (Reliable Connected)'s
in-order-but-one-QP-per-peer model, and vice versa.

## Design
`select_transport()` (`srd_selector.cpp`) takes a peer count, message
size, and whether the caller needs transport-level ordering, and picks
SRD once peer count crosses a threshold (RC's O(N) queue-pair cost starts
dominating) unless the caller needs in-order delivery *and* stays under
that threshold — e.g. ring all-reduce always talks to exactly 2 neighbors
regardless of world size, so it's RC-eligible on peer count alone even at
large scale, while an all-to-all MoE dispatch or the gRPC control plane's
fan-out (step 6) push peer count up and tip the decision toward SRD. The
threshold constant is a placeholder (`kPeerCountThreshold = 8` in
`srd_selector.cpp`) pending the real latency/throughput crossover
measurement this step's Results table is TODO for — the *shape* of the
decision (peer count + ordering requirement) doesn't change once real
numbers replace the constant.

## Sanity-run output (Mac, 2026-07-19)

`srd_selector_test` — five scenarios from this project's own future
transport users (ring all-reduce, a raw two-sided stream, MoE dispatch,
gRPC fan-out, a small ordered cluster):

```
ring all-reduce (2 peers, ordered)            OK  (got RC, want RC)
raw two-sided stream (2 peers, ordered)       OK  (got RC, want RC)
MoE all-to-all dispatch (64 peers, sequenced) OK  (got SRD, want SRD)
gRPC control plane fan-out (16 peers)         OK  (got SRD, want SRD)
small ordered cluster below threshold (4 peers, ordered) OK  (got RC, want RC)
PASS
```

## Results
TODO: measure the real SRD-vs-RC latency/throughput crossover on EFA
hardware and replace `kPeerCountThreshold`'s placeholder value.

| Transport | Peers | Latency (µs) | Throughput (msg/s) | QPs used |
|-----------|-------|--------------|---------------------|----------|
| RC | 2 | TODO | TODO | TODO |
| SRD | 2 | TODO | TODO | TODO |
| RC | 16 | TODO | TODO | TODO |
| SRD | 16 | TODO | TODO | TODO |

## Hardware notes
- `srd_selector` builds and runs anywhere (validated on Mac). Measuring
  the real crossover requires EFA-enabled instances with both transports
  available to compare.
