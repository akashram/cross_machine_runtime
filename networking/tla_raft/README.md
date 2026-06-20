# TLA+ Specification for Raft

**Status: STUB — TLC model checker runs on any machine with Java.**

## What this measures
TLA+ model of the Raft implementation. TLC model checker verifies safety
(at most one leader per term, log matching) and liveness (eventually commits).

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
