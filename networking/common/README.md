# Shared Transport Substrate (`Channel` / `TcpChannel`)

**Status: code-complete AND locally run — like `compiler/cost_model/`, this
has no Linux/EFA-only dependency.**

## What this is
Not a PLAN.md step on its own — it's the shared substrate steps 11-23
(all-reduce variants, the broadcast/reduce-scatter/all-gather library,
topology scheduling, vector clocks, Chandy-Lamport, Raft, backpressure,
hedged requests, multitenancy) are written against, plus the "TCP
baseline" comparison column in `rdma_v1/README.md`. See
`networking/DESIGN.md` for why this exists as its own component instead of
being duplicated per algorithm.

## Design
`Channel` is an abstract blocking send/recv interface keyed by numeric
rank. `TcpChannel` implements it over real POSIX sockets — one persistent
full-duplex TCP connection per rank pair, established by a barrier-like
constructor (accept from every higher rank, connect to every lower rank).
`make_tcp_loopback_mesh(world_size, base_port)` spins up the whole mesh
concurrently (one thread per rank) over `127.0.0.1` so every algorithm test
in this directory tree can validate correctness on a single machine, no
multi-node EFA hardware required. Swapping in a future `RdmaChannel`
(built on `rdma_v1`/`rdma_onesided` once there's Linux+EFA hardware) for
performance validation changes zero algorithm code — only which `Channel`
gets constructed.

## Sanity-run output (Mac, 2026-07-19)

`channel_test`: 4-rank loopback mesh over real TCP sockets, all-pairs
exchange verifying every rank receives exactly its peer's rank id.
Compiled and run directly (`clang++ -std=c++20 -O2 -Wall -Wextra -pthread`,
zero warnings):

```
mesh established: 4 ranks over real TCP sockets (127.0.0.1:34567-34570)
rank 0: all-pairs exchange OK
rank 1: all-pairs exchange OK
rank 2: all-pairs exchange OK
rank 3: all-pairs exchange OK
PASS
```

## Hardware notes
- Builds and runs anywhere (validated on Mac). Real multi-node deployment
  is a one-line change (loopback host list → real hostnames/ports); no
  code in the algorithms consuming `Channel` changes.
