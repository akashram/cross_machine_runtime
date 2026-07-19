# TLA+ Specification for Collective Protocol

**Status: spec written, not yet run through TLC — deliberately deferred, same as tla_raft/.**

## What this measures
TLA+ model verifying that the ring all-reduce protocol is deadlock-free
and produces consistent results across all ranks.

## Design
`Collective.tla` models `ring_allreduce/ring_allreduce.cpp` specifically
around the property that implementation actually depends on: each
adjacent rank pair shares **one bounded-capacity channel** (mirroring
`networking/common::TcpChannel`'s one-bidirectional-socket-per-pair
design — see that component's README), not independent per-direction
channels. Modeling independent channels would prove nothing about the
real implementation, since the whole reason `ring_allreduce.cpp` needs
its even-sends-first/odd-receives-first parity rule is exactly this
sharing (for `N=2`, a rank's left and right neighbor are literally the
same peer). `SendsFirst(r)` encodes that same parity rule as a TLA+
predicate; chunk contents are abstracted to *contributor sets*
(`chunk[r][c] = Ranks` once every rank has contributed) rather than
literal float sums, since set equality is what TLC can check directly.
"No deadlock" isn't a separate invariant — it's TLC's built-in check that
every non-terminal reachable state has an enabled action, which is
exactly what the ordering rule is supposed to guarantee.

**Why this wasn't run this session:** same call as `tla_raft/` — the user
chose to write the specs for real and leave TLC model-checking as TODO
rather than spin up a Java toolchain right now. `ring_allreduce.cpp`
itself *was* built and run locally, repeatedly, including the exact
`N=2` case this spec is about (see `ring_allreduce/README.md`) — this
file is where "verified by running the code" and "verified by model
checking" are deliberately two different tools, not two different levels
of confidence.

## Results
TODO: run TLC.

| Property | TLC result |
|----------|-----------|
| No deadlock | TODO |
| All ranks agree on final value | TODO |

## Hardware notes
- Required: Java (TLC), any OS
