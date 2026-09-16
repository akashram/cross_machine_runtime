# mpi_allreduce -- real MPI ring all-reduce

**Status: code-complete, toolchain-gated, UNRUN -- no MPI implementation
installed on this Mac (ask-before-install declined this session, see
CLAUDE.md's Phase 22 update).**

## What this measures

PLAN.md Phase 22 step 1: reimplement `networking/ring_allreduce`'s
hand-rolled, socket-based ring all-reduce using real MPI (`MPI_Sendrecv`
for both phases), then compare it directly -- correctness and measured
wall-clock time at the same rank count -- against `MPI_Allreduce`, the
vendor-optimized collective every real MPI implementation ships.

## Design

`mpi_ring_allreduce.cpp` reimplements the exact same two-phase algorithm
`networking/ring_allreduce/ring_allreduce.h` documents:

1. **Reduce-scatter** (n-1 rounds): each rank sends its current copy of
   chunk `(rank - step) % world_size` to the next rank in the ring and
   receives the previous rank's contribution to chunk
   `(rank - step - 1) % world_size`, accumulating in place. After n-1
   rounds, rank r holds the fully-reduced chunk `(r+1) % world_size`.
2. **All-gather** (n-1 rounds): circulates each rank's now-correct chunk
   around the ring so every rank ends up with the complete, fully-reduced
   buffer.

Same chunk-ownership convention as the hand-rolled version (chunk `i`
spans `[i*count/N, (i+1)*count/N)`, rank r's phase-1 output is chunk
`(r+1) % world_size`) -- deliberately kept identical so the two
implementations are comparable point-for-point, not just "both compute a
sum."

`MPI_Sendrecv` (not separate non-blocking `MPI_Isend`/`MPI_Irecv`) is
used for both phases -- it performs the send and receive as one atomic
call, sidestepping the same "who sends first" deadlock-avoidance ordering
`ring_allreduce.cpp`'s own header comment flags as a real concern on a
shared-socket transport; MPI guarantees `MPI_Sendrecv` is deadlock-free
for exactly this ring-exchange pattern.

Correctness check: each rank contributes `buf[i] = rank + 1` at every
position, so the correct sum everywhere is the closed-form
`world_size*(world_size+1)/2` -- the same fixed, hand-checkable ground
truth style `ring_allreduce_test.cpp` uses, not a randomized check that
would need a second reduction to validate.

## Toolchain gate

No OpenMPI/MPICH is installed on this Mac. This session ran
`brew info open-mpi` and confirmed a real, current, bottled Homebrew
formula exists (`open-mpi` 5.0.10, no macOS-specific caveats, ordinary
dependency set: `gcc`, `hwloc`, `libevent`, `pmix`, `prrte`) -- so unlike
Slurm (see `slurm_jobs/README.md`), this step's only gate is the standing
no-new-local-installs decision for this session, not a genuine platform
limitation. `hpc_cluster/CMakeLists.txt` gates this subdirectory behind
`find_package(MPI)`, mirroring how root `CMakeLists.txt` gates
`gpu_engine/` behind `check_language(CUDA)`.

## Results

TODO: run once `brew install open-mpi` is approved.

| | Correctness | Wall-clock (4 ranks, 4MB buffer) | Effective GB/s |
|---|---|---|---|
| Hand-rolled ring (`MPI_Sendrecv`) | TODO | TODO | TODO |
| `MPI_Allreduce` (vendor collective) | TODO | TODO | TODO |
| `networking/ring_allreduce` (loopback socket, real captured number) | PASS (see that step's README) | TODO -- fill in from that README | TODO |

## Hardware/toolchain notes

- Required: `brew install open-mpi` (or MPICH), then
  `cmake --preset debug && cmake --build --preset debug --target mpi_ring_allreduce`.
- Run: `mpirun -n 4 ./build/debug/hpc_cluster/mpi_allreduce/mpi_ring_allreduce [count]`.
- Expect `MPI_Allreduce` to beat the hand-rolled ring at small rank counts
  (vendor collectives use algorithm selection -- often a Rabenseifner-style
  reduce-scatter+allgather at large messages, similar in shape to the ring
  algorithm here, but with additional tuning: e.g. switching to a different
  algorithm below a message-size threshold). Whether the hand-rolled ring
  is competitive at the buffer size tested is exactly what running this
  will show, not assumed in advance.
