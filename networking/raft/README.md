# Raft Consensus

**Status: STUB — requires Linux (any single machine for initial testing).**

## What this measures
Raft consensus from scratch: leader election, log replication, membership changes.
Validated against TLA+ spec (tla_raft/).

## Results
TODO: implement and run on Linux.

| Scenario | Election time (ms) | Log replication latency (ms) | Throughput (ops/s) |
|----------|-------------------|-----------------------------|--------------------|
| 3-node cluster, no failures | TODO | TODO | TODO |
| Leader failure + reelection | TODO | TODO | TODO |
| Network partition (minority) | TODO | TODO | TODO |

## Hardware notes
- Required: Linux (can simulate multi-node on single machine with processes)
