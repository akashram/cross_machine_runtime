# training_worker — real process-per-rank distributed training driver

**Status: code-complete AND locally run — real inter-process TCP, 4
separate OS processes on this Mac, no CUDA/Linux dependency.**

## What this closes

The Phase 16 container/orchestration work (`k8s/training/statefulset.yaml`)
disclosed a real gap: every multi-rank step in this repo (canonically
`distributed_training/full_training_loop/training_loop_test.cpp`)
simulates ranks as `std::async` tasks inside ONE process, over a loopback
TCP mesh (`netcommon::make_tcp_loopback_mesh`) where every rank shares a
single hostname (`127.0.0.1`). A Kubernetes `StatefulSet` needs one real
OS **process** per pod, each with its own DNS name
(`<name>-<ordinal>.<headless-svc>`) — threads-in-one-process doesn't map
onto that at all.

`training_worker.cpp` is that real driver:
- Reads `RANK`, `WORLD_SIZE`, `PEER_HOSTS` (comma-separated, one hostname
  per rank) from the environment — the standard convention this repo's
  own gap report noted was absent (confirmed via `grep -r RANK` before
  writing this: zero hits anywhere in the tree beforehand).
- Builds exactly ONE `netcommon::TcpChannel` for this rank, via a new
  constructor overload added to `networking/common/channel.{h,cpp}`
  (`TcpChannel(rank, world_size, base_port, peer_hosts)` — see that
  file's own doc comment). The existing single-`host`-string constructor
  is untouched, byte-for-byte, so every existing caller/test keeps
  working exactly as before (`channel_test` still passes — see below).
  The new constructor binds `0.0.0.0` on the listen side (a rank's own
  listen address isn't necessarily reachable at the name its peers dial
  it by) and dials each lower rank at `peer_hosts[j]` instead of one
  shared address.
- Runs the IDENTICAL training math `training_loop_test.cpp`'s per-rank
  lambda body runs — same `MLP`, same `ZeroStage1Optimizer`, same
  `global_grad_norm`/`clip_grad_by_global_norm`, same `ring_allreduce`,
  same `AsyncCheckpointWriter` — reused unchanged via the same headers.
  `training_loop_test.cpp` itself is untouched (still an independently
  passing test); this file duplicates only the ~15-line toy-dataset
  generator (deterministic given a fixed seed, so every process
  independently reconstructs the identical dataset/shard boundaries/
  initial weights with zero extra coordination messages), not any actual
  training logic.

## New Channel constructor (`networking/common/channel.{h,cpp}`)

Added a `channel_peerhosts_test.cpp` sanity check (3 ranks, threads over
loopback, same style as the existing `channel_test.cpp`) that exercises
the new constructor path in isolation before `training_worker` depends on
it for real:

```
peer_hosts mesh established: 3 ranks, each dialed by hostname
rank 0: all-pairs exchange OK
rank 1: all-pairs exchange OK
rank 2: all-pairs exchange OK
PASS
```

The existing `channel_test` (4-rank loopback mesh via the untouched
single-`host` constructor) still passes unchanged:

```
mesh established: 4 ranks over real TCP sockets (127.0.0.1:34567-34570)
rank 0: all-pairs exchange OK
rank 1: all-pairs exchange OK
rank 2: all-pairs exchange OK
rank 3: all-pairs exchange OK
PASS
```

## Real multi-process run (captured 2026-07-30, this Mac)

`run_local_demo.sh` launches 4 REAL, separate `training_worker` OS
processes on `127.0.0.1` (distinct ports, `PEER_HOSTS` all pointing at
`127.0.0.1` since this is a single-machine demo — a real K8s deployment
would set each pod's actual per-ordinal DNS name instead) and waits for
all of them:

```
$ ./distributed_training/training_worker/run_local_demo.sh
training_worker real multi-process demo: world_size=4 ckpt_dir=/tmp/...
--- rank 0 log ---
training_worker: rank=0 world_size=4 base_port=46301 epochs=30 ckpt_dir=/tmp/...
rank 0: connected to all 4 peers over real TCP
rank 0: training loss 2.6317 -> 0.0008
rank 0 phase breakdown (30 steps): forward_backward=12.106ms grad_clip=5.345ms grad_sync=10.523ms optimizer_step=15.977ms checkpoint=0.304ms total=44.257ms
rank 0: PASS
--- rank 1 log ---
rank 1: connected to all 4 peers over real TCP
rank 1 phase breakdown (30 steps): forward_backward=13.153ms grad_clip=32.637ms grad_sync=9.847ms optimizer_step=7.974ms checkpoint=0.134ms total=63.745ms
rank 1: PASS
--- rank 2 log ---
rank 2: connected to all 4 peers over real TCP
rank 2 phase breakdown (30 steps): forward_backward=12.398ms grad_clip=28.209ms grad_sync=9.910ms optimizer_step=15.755ms checkpoint=0.389ms total=66.661ms
rank 2: PASS
--- rank 3 log ---
rank 3: connected to all 4 peers over real TCP
rank 3 phase breakdown (30 steps): forward_backward=12.066ms grad_clip=31.928ms grad_sync=10.625ms optimizer_step=10.860ms checkpoint=0.702ms total=66.181ms
rank 3: PASS
```

**Loss trajectory matches `full_training_loop/README.md`'s thread-
simulated baseline exactly: `2.6317 -> 0.0008` both ways.** That's a real,
useful correctness check, not a coincidence — both use the same fixed
RNG seeds for the initial model and dataset, so bit-identical training
math run over real inter-process TCP instead of intra-process threads
should (and does) reproduce the identical loss curve. If the two had
diverged, that would have pointed at a real bug in either the new
`peer_hosts` Channel path or an accidental behavior difference between
the process-per-rank and thread-per-rank drivers.

Checkpoint files are real, per-rank, per-epoch shards written by 4
independent processes to a shared directory (`CHECKPOINT_DIR`), inspected
directly during development (not via the demo script, which cleans up
after itself):

```
$ ls $CHECKPOINT_DIR
rank0_epoch9.bin  rank0_epoch19.bin  rank0_epoch29.bin
rank1_epoch9.bin  rank1_epoch19.bin  rank1_epoch29.bin
rank2_epoch9.bin  rank2_epoch19.bin  rank2_epoch29.bin
rank3_epoch9.bin  rank3_epoch19.bin  rank3_epoch29.bin
```

Per-rank phase timings are noisier than `full_training_loop`'s
thread-based numbers (compare `grad_clip`: 5.3ms on rank 0 vs. ~30ms on
ranks 1-3) — expected: real separate processes are scheduled
independently by the OS, whereas the thread-based version's phases are
measured inside one process's cooperative `std::async` scheduling. This
is the real overhead difference process-per-rank execution introduces
that thread-simulation hides, not a bug.

## k8s/training/statefulset.yaml

Updated to reference this real binary. `RANK` comes from parsing the pod
hostname's trailing ordinal (the standard StatefulSet
`<name>-<ordinal>` pattern), `WORLD_SIZE` from the replica count,
`PEER_HOSTS` from the predictable per-pod DNS pattern under the headless
service (`<name>-0.<svc>,<name>-1.<svc>,...`). Still cluster-unrun (no
real Kubernetes cluster on this Mac — see `containers/README.md`), but it
now points at a real, tested (locally, multi-process) binary instead of
an aspirational one.

## Scope notes
- Toy scale only (same `MLP({2,16,3}, ...)` and 120-sample synthetic
  dataset every other Phase 6 step uses) — this validates the
  process-per-rank mechanism, not a production-scale training run.
- `PEER_HOSTS` all being `127.0.0.1` in the local demo means this hasn't
  validated cross-*machine* TCP, only cross-*process* TCP with distinct
  ports — the K8s deployment (once a real cluster exists) is what
  validates true cross-host communication; the mechanism (dialing a
  provided hostname per peer rather than one shared address) is
  identical either way.
