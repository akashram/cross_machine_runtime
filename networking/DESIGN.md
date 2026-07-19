# Distributed Layer + Networking Architecture

Status: Phase 5 code-complete (25/25 steps). Unlike Phase 3 (GPU, 0% run)
and Phase 4 (MLIR, one component run), **12 of Phase 5's 25 steps are
locally built and run** — `common/`, `rdma_v1`'s TCP baseline, `efa_srd`,
`ring_allreduce`, `halving_doubling`, `tree_allreduce`, `collectives`,
`topo_scheduler`, `vector_clocks`, `chandy_lamport`, `raft`,
`backpressure`, `hedged_requests`, `multitenancy` — because most of
distributed systems theory needs a network, not EFA specifically. This
document covers the decisions behind that split and the handful of real
bugs caught by actually running things instead of just writing them.

---

## 1. Why a shared `Channel` abstraction instead of writing each
   algorithm against sockets directly

Every algorithm from ring all-reduce through multi-tenancy is written
against `networking/common::Channel`, not `libfabric`/EFA directly — those
need 2+ EFA-enabled nodes this project doesn't have. `TcpChannel` is a
real, portable (POSIX sockets) implementation good enough to validate
every algorithm's *correctness* right now; swapping in an `RdmaChannel`
(built on `rdma_v1`/`rdma_onesided` once there's Linux+EFA hardware) for
*performance* validation later changes zero algorithm code. This is the
single decision that turned Phase 5 from "16 more write-and-hope
components like Phase 3/4" into "12 components with real captured test
output" — see `common/README.md`.

## 2. `TcpChannel` shares one bidirectional socket per rank pair — and
   that's not a simplification, it's what created the interesting bugs

The alternative considered: one socket per *direction* per pair (send
socket + separate recv socket). Real RDMA queue pairs and most production
RPC frameworks are architecturally closer to the shared-bidirectional
model, so this wasn't purely a convenience choice — but it does mean
`ring_allreduce`, `halving_doubling`, and `raft` all had to reckon with a
real ordering hazard: two ranks that are each other's *only* neighbor
(`world_size == 2`, or any node talking to its sole peer) can deadlock if
both sides try to send before either receives, once a message exceeds the
kernel socket buffer. The fix — even ranks send-first, odd ranks
receive-first — is genuinely the standard technique (see
`ring_allreduce.cpp`), but it only became a *known requirement* here
because the local test suite could actually trigger it. `tla_collective`
(step 25) formalizes exactly this property against the real channel-
sharing model, not an idealized independent-channels version.

## 3. Chunk ownership in the ring algorithms is `(rank+1) % N`, not `rank`
   — and getting this wrong was a real, caught bug

`ring_reduce_scatter`'s output convention (which chunk index ends up
holding the correct reduced value for a given rank) falls out of which
direction the ring sends in — it's `(rank+1) % world_size`, not `rank`.
`collectives::ReduceScatter`/`AllGather` first shipped assuming `rank`,
and the local test caught it immediately (`collectives/README.md` has the
full story). This is exactly the kind of off-by-one that's cheap to catch
against a loopback mesh and expensive to debug against real EFA hardware
at 2am — the concrete argument for building `common/Channel` before
writing any algorithm against it, not after.

## 4. `advanceCommitIndex`'s same-term-only rule is the one line in
   `raft.cpp` most worth reading twice

A leader must never commit a previous-term log entry purely by counting
replicas — only entries from its own current term, with older entries
committed as a side effect of a later same-term entry committing (Raft
paper §5.4.2). Skipping this check is *almost always* safe, which is
exactly what makes it the most common real-world Raft bug: most test
scenarios (including, initially, this project's) never hit the specific
leader-crash/re-election interleaving where it matters. It's checked
explicitly in `raft.cpp`, cross-referenced against `tla_raft/Raft.tla`'s
`OneLeaderPerTerm`/`LogMatching` invariants (not yet run through TLC —
see that step's README for why), and called out again here because
"almost always safe" bugs are the ones worth over-documenting.

## 5. Two shutdown-coordination bugs, one shared root cause

Both `chandy_lamport::ChandyLamportNode::stop()` and
`raft::RaftNode::stop()` initially tried to cleanly join their per-peer
receiver threads, each of which is blocked in `Channel::recv` — which
only unblocks when the peer sends something. `chandy_lamport`'s test
originally called `stop()` on each node sequentially, deadlocking (node 0
waits for node 1 to send a shutdown frame, but node 1's `stop()` hasn't
run yet); fixed by stopping all nodes concurrently. `raft_test.cpp`'s
leader-failover scenario has no fix *available* at the call-site level —
survivors keep running while the "crashed" node stops, so it will never
reciprocate a shutdown handshake. Fixed at the class level instead:
`RaftNode::stop()` detaches its receiver threads rather than joining them
(safe here because `Channel` outlives every `RaftNode` that references
it — see `raft.cpp`'s `stop()` comment). Same underlying lesson both
times: a clean-shutdown protocol that depends on peer cooperation is only
as available as flakiness-free peer cooperation.

## 6. What's still write-and-hope (the other 13 steps)

`efa_setup`, `rdma_v1`'s EFA half, `rdma_onesided`, `ptp`, `grpc_control`,
`flatbuffers_data`, `af_xdp`, `userspace_net`, `nic_deep_dive`,
`nccl_tuning`, `chaos`, plus the two TLA+ specs, are real, complete code
that needs either Linux-only kernel APIs (AF_XDP, PTP hardware
timestamping), a specific NIC/driver (EFA, XDP support), external
libraries not installed here (gRPC, FlatBuffers), GPU hardware (NCCL), or
a Java toolchain (TLC) — same "write it for real, validate on the
hardware/tooling it actually needs" stance as Phases 3 and 4. See each
step's own README for its specific gate.
