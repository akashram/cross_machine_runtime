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

**A real bug caught locally**, not just a design note: `stop()` originally
sent a shutdown frame to every peer and then *joined* its receiver
threads, waiting for each peer to reciprocate. That works for stopping an
entire cluster at once, but deadlocks the leader-failover test — a
"crashed" node's peers are still running and never send it a shutdown
frame back. Fixed by detaching receiver threads instead of joining them
(`Channel::recv` has no cancellation primitive short of closing the
socket, which isn't this class's to close): the ticker thread still joins
cleanly since it only depends on its own `running_` flag, not peer
cooperation.

## Sanity-run output (Mac, loopback, 2026-07-19)

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
  full failover). Required: Linux for the real multi-node/multi-process
  timing numbers above.
