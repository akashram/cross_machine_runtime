# tlc

**Status: code-complete, toolchain-gated — real TLC model configs, unrun.
No Java runtime on this Mac; the user asked to revisit installing one
before hardware provisioning rather than deciding now (same treatment as
Phase 8's JAX decision — see this repo's memory).**

## What this measures

PLAN.md Phase 10 step 4: run TLC on all TLA+ specs (Raft, all-reduce
protocol), verify no violations under all interleavings.

## Design

Both `networking/tla_raft/Raft.tla` and
`networking/tla_collective/Collective.tla` were written and checked in
without ever being run through TLC (no Java toolchain used in those
sessions either — see their own READMEs). This step is what actually
runs them:

- `RaftMC.tla`: a model-checking wrapper `EXTENDS Raft` rather than an
  edit to `Raft.tla` itself — the standard TLA+ pattern for keeping a
  spec free of model-specific bounding. Needed because `Timeout` has no
  upper bound on `currentTerm` and `ClientRequest` has no upper bound on
  log length (both correct for the real, unbounded spec — a real cluster
  runs forever), so without SOME bound TLC's state space is infinite.
  `StateConstraint` bounds both to 3 — small enough to terminate, large
  enough to still reach multiple election terms and multi-entry logs
  (bounding tighter risks silently making `OneLeaderPerTerm`'s
  cross-term case unreachable, proving nothing about it).
- `RaftMC.cfg`: `Server = {s1,s2,s3}`, `Value = {v1,v2}`, checks all
  three safety invariants (`OneLeaderPerTerm`, `LogMatching`,
  `CommittedEntriesAgree`) from `Raft.tla`. `CHECK_DEADLOCK` left at
  TLC's default (`TRUE`) — Raft has no legitimate "nothing left to do"
  state (a Leader can always `SendAppendEntries`, a Follower/Candidate
  can always `Timeout`), so a reported deadlock here would mean a real
  spec bug.
- `Collective.cfg`: `N = 3` (small enough for exhaustive search, large
  enough that `Right(r) != Left(r)` for every rank — the `N=2` case
  collapses both ring neighbors onto the same channel, a valid but more
  degenerate case worth checking separately later). Checks
  `AllRanksAgree`. `CHECK_DEADLOCK FALSE` — unlike Raft, `AllFinished`
  IS a legitimate terminal state here (every rank completed all
  `2*(N-1)` rounds), so without this TLC would report every run's own
  successful completion as a false-positive deadlock.
- `run_tlc.sh`: copies each spec's `.tla` (and MC wrapper/`.cfg`, for
  Raft) into a scratch directory before invoking TLC, rather than
  depending on a specific TLC library-search-path flag to resolve
  `EXTENDS Raft` — correct regardless of TLC version.

## Results
TODO: run via `TLA_JAR=/path/to/tla2tools.jar ./run_tlc.sh` once a Java
runtime is installed (`tla2tools.jar`:
github.com/tlaplus/tlaplus/releases).

| Spec | Invariant | Result | States explored | Time |
|---|---|---|---|---|
| Raft (3 servers, 2 values) | OneLeaderPerTerm | TODO | TODO | TODO |
| Raft (3 servers, 2 values) | LogMatching | TODO | TODO | TODO |
| Raft (3 servers, 2 values) | CommittedEntriesAgree | TODO | TODO | TODO |
| Collective (N=3) | AllRanksAgree | TODO | TODO | TODO |

## Hardware notes
- Required: Java runtime (any OS — TLC itself has no Linux/hardware
  dependency, unlike most of this repo's other gated steps), plus
  `tla2tools.jar`.
