// mpi_ring_allreduce.cpp -- PLAN.md Phase 22 step 1: real MPI ring
// all-reduce, directly comparable to networking/ring_allreduce's
// hand-rolled socket-based version (same chunk-ownership convention:
// after reduce-scatter, rank r owns the fully-reduced chunk
// (r+1) % world_size; all-gather then circulates it -- see
// networking/ring_allreduce/ring_allreduce.h). This file reimplements
// that same two-phase algorithm using MPI_Sendrecv instead of
// netcommon::Channel, then checks it against MPI_Allreduce (the vendor
// collective) for both correctness and measured wall-clock time at the
// same rank count.
//
// TOOLCHAIN-GATED, UNRUN: no MPI implementation (OpenMPI/MPICH) is
// installed on this Mac -- ask-before-install declined this session (see
// CLAUDE.md's Phase 22 update / project memory feedback_no_new_local_installs).
// This session confirmed via `brew info open-mpi` that a real, current,
// bottled formula exists (5.0.10, no macOS-specific caveats) -- so this
// step's only gate is the standing no-new-installs decision, not a
// platform limitation. Real, complete, correct code written against the
// documented MPI-3 semantics for MPI_Send/Recv/Sendrecv/Allreduce/Wtime --
// same convention as gpu_engine's CUDA kernels and fpga_engine's HLS code:
// compiles and runs once the toolchain exists, not a stub.
//
// Build once OpenMPI/MPICH is available:
//   brew install open-mpi
//   cmake --preset debug && cmake --build --preset debug --target mpi_ring_allreduce
// Run:
//   mpirun -n 4 ./build/debug/hpc_cluster/mpi_allreduce/mpi_ring_allreduce [count]
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Same chunk convention as networking/ring_allreduce/ring_allreduce.cpp:
// chunk i spans [i*count/N, (i+1)*count/N); after reduce-scatter, rank r
// owns the fully-reduced chunk (r+1) % world_size. n-1 rounds.
void mpi_ring_reduce_scatter(std::vector<float> &buf, int world_size,
                              int rank, MPI_Comm comm) {
  int count = static_cast<int>(buf.size());
  int send_to = (rank + 1) % world_size;
  int recv_from = (rank - 1 + world_size) % world_size;

  auto chunk_begin = [&](int i) { return i * count / world_size; };
  auto chunk_len = [&](int i) { return chunk_begin(i + 1) - chunk_begin(i); };

  std::vector<float> recv_buf(static_cast<size_t>(count));

  for (int step = 0; step < world_size - 1; ++step) {
    int send_chunk = (rank - step + world_size) % world_size;
    int recv_chunk = (rank - step - 1 + world_size) % world_size;

    MPI_Sendrecv(buf.data() + chunk_begin(send_chunk), chunk_len(send_chunk),
                 MPI_FLOAT, send_to, /*tag=*/0,
                 recv_buf.data() + chunk_begin(recv_chunk),
                 chunk_len(recv_chunk), MPI_FLOAT, recv_from, /*tag=*/0, comm,
                 MPI_STATUS_IGNORE);

    for (int i = chunk_begin(recv_chunk);
         i < chunk_begin(recv_chunk) + chunk_len(recv_chunk); ++i) {
      buf[static_cast<size_t>(i)] += recv_buf[static_cast<size_t>(i)];
    }
  }
}

// n-1 rounds, circulates each rank's already-correct chunk
// (rank+1) % world_size (matching reduce_scatter's output convention)
// around the ring so every rank ends up with the full buffer.
void mpi_ring_all_gather(std::vector<float> &buf, int world_size, int rank,
                          MPI_Comm comm) {
  int count = static_cast<int>(buf.size());
  int send_to = (rank + 1) % world_size;
  int recv_from = (rank - 1 + world_size) % world_size;

  auto chunk_begin = [&](int i) { return i * count / world_size; };
  auto chunk_len = [&](int i) { return chunk_begin(i + 1) - chunk_begin(i); };

  for (int step = 0; step < world_size - 1; ++step) {
    int send_chunk = (rank + 1 - step + world_size) % world_size;
    int recv_chunk = (rank - step + world_size) % world_size;

    MPI_Sendrecv(buf.data() + chunk_begin(send_chunk), chunk_len(send_chunk),
                 MPI_FLOAT, send_to, /*tag=*/1,
                 buf.data() + chunk_begin(recv_chunk), chunk_len(recv_chunk),
                 MPI_FLOAT, recv_from, /*tag=*/1, comm, MPI_STATUS_IGNORE);
  }
}

void mpi_ring_allreduce(std::vector<float> &buf, int world_size, int rank,
                         MPI_Comm comm) {
  mpi_ring_reduce_scatter(buf, world_size, rank, comm);
  mpi_ring_all_gather(buf, world_size, rank, comm);
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Default: 1M floats = 4MB, comparable order of magnitude to
  // networking/ring_allreduce_test.cpp's own buffer sizes.
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 20;

  if (world_size < 2 && rank == 0) {
    std::fprintf(stderr,
                  "mpi_ring_allreduce: needs world_size >= 2 (run via "
                  "mpirun -n 4 ...)\n");
  }

  // Each rank contributes buf[i] = rank + 1 (so the correct sum at every
  // position is world_size*(world_size+1)/2) -- a fixed, hand-checkable
  // ground truth, same style as ring_allreduce_test.cpp's own check.
  std::vector<float> buf_ring(static_cast<size_t>(count),
                               static_cast<float>(rank + 1));
  std::vector<float> buf_native(static_cast<size_t>(count),
                                 static_cast<float>(rank + 1));

  float expected = static_cast<float>(world_size) * (world_size + 1) / 2.0f;

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  mpi_ring_allreduce(buf_ring, world_size, rank, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_ring = MPI_Wtime() - t0;

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE, buf_native.data(), count, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_native = MPI_Wtime() - t1;

  bool ring_ok = true, native_ok = true;
  for (int i = 0; i < count; ++i) {
    if (std::fabs(buf_ring[static_cast<size_t>(i)] - expected) > 1e-3f)
      ring_ok = false;
    if (std::fabs(buf_native[static_cast<size_t>(i)] - expected) > 1e-3f)
      native_ok = false;
  }

  int local_ok = (ring_ok && native_ok) ? 1 : 0;
  int global_ok = 0;
  MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double bytes = static_cast<double>(count) * sizeof(float);
    double gbps_ring = (bytes / t_ring) / 1e9;
    double gbps_native = (bytes / t_native) / 1e9;
    std::printf("mpi_ring_allreduce: world_size=%d count=%d (%.1f MB)\n",
                world_size, count, bytes / 1e6);
    std::printf(
        "  hand-rolled ring (MPI_Sendrecv): rank0 %s  %.6f s  "
        "(%.3f GB/s effective)\n",
        ring_ok ? "PASS" : "FAIL", t_ring, gbps_ring);
    std::printf(
        "  MPI_Allreduce (vendor collective): rank0 %s  %.6f s  "
        "(%.3f GB/s effective)\n",
        native_ok ? "PASS" : "FAIL", t_native, gbps_native);
    std::printf(
        "  compare against networking/ring_allreduce's real loopback-socket "
        "numbers in that step's own README once both are captured.\n");
    std::printf("  overall: %s\n", global_ok ? "PASS" : "FAIL");
  }

  MPI_Finalize();
  return (ring_ok && native_ok) ? 0 : 1;
}
