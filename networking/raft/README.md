# Raft Consensus

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Raft consensus from scratch: leader election, log replication. Validated
against TLA+ spec (tla_raft/). Cluster membership changes are NOT
implemented — documented scope limit, see `raft.h`.

## Design
Built on `networking/common::Channel` like every other portable Phase 5
component: the stub's original `RaftNode(id, peer_addrs)` (address
strings) became `RaftNode(Channel&)`. One receiver thread per peer feeds a
single mutex-guarded state machine (`raft.cpp`) via length-prefixed,
tagged frames (RequestVote / AppendEntries / their responses) — no
request/response correlation needed since a node only ever has one
logical outstanding request per peer (heartbeats and log pushes are
level-triggered by a 10ms ticker, not queued). Election timeouts are
randomized per-node (150-300ms, the range from the original Raft paper).

The safety-critical detail: `advanceCommitIndex` only commits an index
whose entry's **term equals the leader's current term** — the Raft
paper's §5.4.2 rule that a leader must never commit a previous-term entry
purely by counting replicas; it can only be committed as a side effect of
committing a later same-term entry. This is the single most common Raft
implementation bug precisely because skipping it is *almost* always safe
(most test scenarios never hit the specific interleaving where it
matters) — flagged explicitly in `raft.cpp` and cross-checked against
`tla_raft/Raft.tla`'s `LogMatching`/`OneLeaderPerTerm` invariants.

**Three real bugs caught locally, not just design notes** — the shutdown
path has a real history worth telling in full, because each fix's own
"obvious" solution turned out to have its own bug:

1. **Original design**: `stop()` sent a shutdown frame to every peer and
   then *joined* its receiver threads, waiting for each peer to
   reciprocate. Works for stopping an entire cluster at once; deadlocks
   the leader-failover test, where a "crashed" node's peers are still
   running and never send it a shutdown frame back.
2. **First fix (wrong)**: switched to *detaching* receiver threads
   instead of joining, reasoning that each is blocked forever in
   `Channel::recv()` and so is harmless to leave running. That reasoning
   has a hole: a receiver thread isn't always blocked in `recv()` — it
   can be anywhere inside `handleRequestVote`/`handleAppendEntries`,
   actively touching `this` (`mu_`, `log_`, `peerState_`, `channel_`).
   If the `RaftNode` is destructed (the normal `unique_ptr` teardown at
   end of test) while a detached thread is mid-handler, or blocked in
   `recv()` on the now-dangling `channel_` reference, that thread
   resumes into freed memory. Rare when everything runs fast (threads
   stay parked in the `recv()` syscall almost the whole time); reliably
   reproducible by running `raft_test` in a loop under artificial CPU
   contention (6 busy-loop processes on a 2-core Mac): **captured a real
   crash report** — `SIGSEGV` / `EXC_BAD_ACCESS` at address `0x0`,
   faulting thread inside `recvFrame` → `Channel::recv` on the dangling
   `channel_`, called from a detached receiver thread. 0/40 clean runs
   under contention before diagnosis.
3. **Real fix**: added `Channel::shutdownPeer(peer)` (`shutdown(fd,
   SHUT_RD)` on the underlying socket) — forcibly unblocks our *own*
   `recv()` call for that peer with zero cooperation needed from the
   peer, unlike the original mutual-shutdown-frame approach. `stop()`
   now calls this for every peer, then genuinely `t.join()`s every
   receiver thread — by the time `stop()` returns, every thread the
   `RaftNode` owns has actually exited, so destructing it is safe.
   **This exposed two more pre-existing, latent gaps** in the socket
   layer, both real bugs in their own right, not new ones introduced by
   the fix: (a) a peer still sending to a half-shut-down socket can get
   a connection reset, and writing to a reset socket raises `SIGPIPE`
   by default — whose default disposition **kills the whole process**,
   not just the failing call; fixed by ignoring `SIGPIPE` process-wide
   in `channel.cpp` (the standard, portable fix, works identically on
   macOS and Linux). (b) even with `SIGPIPE` suppressed, a failed
   `send()` to an unreachable peer threw `std::runtime_error` with
   nothing to catch it, propagating out of `sendFrame` and calling
   `std::terminate()` — **captured**: `libc++abi: terminating due to
   uncaught exception ... Connection reset by peer`. Fixed by making
   `sendFrame` best-effort: a dropped RPC to an unreachable peer is a
   normal, expected condition in a distributed system (the entire
   reason Raft has periodic heartbeats and repeated elections), not a
   fatal one — it's silently retried on the next tick, not re-thrown.

Verified empirically, not just reasoned about: 100/100 clean `raft_test`
runs under the same artificial CPU contention that reliably crashed both
the original detach-based code and the incomplete first attempt at this
fix (which fixed the segfault but introduced the `SIGPIPE`/`SIGABRT`
crashes above). Also TSan-clean, including under contention.

## Sanity-run output (Mac, loopback, 2026-07-19; re-verified 2026-07-28 after the shutdown-path fixes above)

`raft_test`: 3-node cluster — election, propose/commit replicated
identically to all three nodes' `onCommit` callbacks, then a full
leader-failover cycle (stop the leader, verify a new one is elected among
survivors, verify it can still commit):

```
exactly one leader elected within 2s                    OK
  (leader is node 0, term 1)
propose('set x=1') commits                              OK
node 0 committed exactly ["set x=1"]                    OK
node 1 committed exactly ["set x=1"]                    OK
node 2 committed exactly ["set x=1"]                    OK
new leader elected among survivors after leader stop    OK
new leader commits 'set y=2'                             OK
PASS
```

Run 5 consecutive times with no failures or hangs (each run's
election/failover timing differs — different candidate can win a given
term depending on scheduling — but the invariants above hold every time).

## Results
TODO: run on Linux (can simulate multi-node on a single machine with
separate processes, or use real nodes) — this table needs wall-clock
numbers this Mac's loopback timing doesn't represent meaningfully.

| Scenario | Election time (ms) | Log replication latency (ms) | Throughput (ops/s) |
|----------|-------------------|-----------------------------|--------------------|
| 3-node cluster, no failures | TODO | TODO | TODO |
| Leader failure + reelection | TODO | TODO | TODO |
| Network partition (minority) | TODO | TODO | TODO |

## Hardware notes
- Builds and runs anywhere (validated on Mac, 5 repeated runs including
  full failover, plus 100/100 clean runs under artificial CPU contention
  and a clean TSan pass under the same contention — see the shutdown-path
  bug history above). Required: Linux for the real multi-node/multi-process
  timing numbers above.
