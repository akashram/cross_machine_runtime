# TLA+ Specification for Raft

**Status: spec written, not yet run through TLC — no Java/TLA+ toolchain
used in this session (deliberately deferred; see below).**

## What this measures
TLA+ model of the Raft implementation. TLC model checker verifies safety
(at most one leader per term, log matching) and liveness (eventually commits).

## Design
`Raft.tla` follows the structure of the canonical community Raft TLA+
spec (Ongaro's thesis appendix), scoped down to match exactly what
`raft/raft.cpp` implements: leader election + log replication, no
membership changes, no snapshotting. State variables mirror the C++ class
member-for-member (`currentTerm`, `state`, `votedFor`, `log`,
`commitIndex`, plus leader-only `nextIndex`/`matchIndex`); RPCs are
modeled as records in an in-flight `messages` set rather than actual
network sends, standard for this kind of spec. `AdvanceCommitIndex` only
commits an index whose entry's term equals the leader's *current* term —
deliberately mirroring `raft.cpp`'s `advanceCommitIndex`, including the
comment explaining why that specific check is easy to get wrong.

Three safety invariants (`OneLeaderPerTerm`, `LogMatching`,
`CommittedEntriesAgree`) are stated as direct state predicates TLC can
check against the full reachable-state graph. `EventuallyCommits`
(liveness) is included for completeness but needs weak fairness
(`WF_raftVars(Next)`) added to `Spec` to be meaningful — without it, TLC
can trivially violate it by never scheduling message delivery. Model-size
note for whoever runs this: `Server` and `Value` are both unbounded in the
spec; TLC needs finite instantiations (e.g. `Server = {s1, s2, s3}`,
`Value = {v1, v2}`) via a `.cfg` file, not written yet.

**Why this wasn't run this session:** running TLC requires a Java
toolchain; the user explicitly chose "write the specs, leave
model-checking as TODO" over spinning that up right now (same call as for
`tla_collective/`). The `.cpp` implementation this spec models *was*
locally built and run (`raft/README.md`) — this file is the one place in
Phase 5 where "write it for real" and "verify it for real" are split
across two different tools, not because of hardware, but because of that
choice.

## Results
TODO: run TLC model checker.

| Property | TLC result | States explored | Time |
|----------|-----------|----------------|------|
| Safety: OneLeaderPerTerm | TODO | TODO | TODO |
| Safety: LogMatching | TODO | TODO | TODO |
| Liveness: EventuallyCommits | TODO | TODO | TODO |

## Hardware notes
- Required: Java (for TLC), any OS
- Tool: TLA+ toolbox or `tla2tools.jar`
