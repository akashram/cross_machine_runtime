//===- channel.h - Portable point-to-point transport shared by Phase 5 --===//
//
// Every algorithm from step 11 onward (ring/halving-doubling/tree
// all-reduce, the broadcast/reduce-scatter/all-gather library, vector
// clocks, Chandy-Lamport, Raft, backpressure, hedged requests,
// multitenancy) is written against this abstract `Channel`, not against
// libfabric/EFA directly — those steps need 2+ EFA-enabled nodes this
// project doesn't have yet. `TcpChannel` is a real, portable (POSIX
// sockets — builds and runs on macOS and Linux) implementation good
// enough to validate every algorithm's *correctness* right now; swapping
// in an RdmaChannel (rdma_v1/rdma_onesided, once built on Linux+EFA) for
// *performance* validation later changes zero algorithm code. See
// networking/DESIGN.md for the full rationale.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace netcommon {

// Blocking, reliable, ordered point-to-point transport between `world_size`
// numbered ranks. `send`/`recv` transfer exactly `len` bytes or throw
// std::runtime_error — callers (all of them fixed-message-size collective
// algorithms) always know the exact size up front, so there's no framing
// header on the wire.
class Channel {
public:
  virtual ~Channel() = default;
  virtual void send(int dst_rank, const void *data, size_t len) = 0;
  virtual void recv(int src_rank, void *data, size_t len) = 0;
  virtual int rank() const = 0;
  virtual int world_size() const = 0;
};

// Real TCP, one persistent full-duplex socket per rank pair. Constructing
// a TcpChannel blocks until its rank's slice of the full mesh (accept from
// every higher rank, connect to every lower rank) is established — the
// constructor doubles as a barrier, so `make_tcp_loopback_mesh` below can
// return with every channel already connected to every other.
//
// Deadlock note: because each pair shares one bidirectional socket, code
// that does `send` then `recv` on *both* ends of a pair will deadlock once
// messages exceed the kernel socket buffer (each side blocks in `send`
// waiting for the other to `recv`, and neither does). Every algorithm in
// this directory tree that talks to a fixed partner (ring all-reduce's
// neighbor exchange) alternates order by rank parity to avoid this — see
// ring_allreduce.cpp for the canonical pattern.
class TcpChannel : public Channel {
public:
  TcpChannel(int rank, int world_size, uint16_t base_port,
             const std::string &host = "127.0.0.1");
  ~TcpChannel() override;

  TcpChannel(const TcpChannel &) = delete;
  TcpChannel &operator=(const TcpChannel &) = delete;

  void send(int dst_rank, const void *data, size_t len) override;
  void recv(int src_rank, void *data, size_t len) override;
  int rank() const override { return rank_; }
  int world_size() const override { return world_size_; }

private:
  int rank_;
  int world_size_;
  std::vector<int> fds_; // fds_[peer] == shared bidirectional socket to `peer`
};

// Spawns `world_size` TcpChannels connected to each other over 127.0.0.1
// (ports base_port..base_port+world_size-1), one per thread so the mesh
// handshake (each rank both accepts and connects) can proceed concurrently
// instead of deadlocking on ordering. Returns once every channel is fully
// connected. This is the harness every locally-run algorithm test in this
// directory tree uses in place of real multi-node EFA hardware.
std::vector<std::unique_ptr<Channel>> make_tcp_loopback_mesh(int world_size,
                                                               uint16_t base_port);

} // namespace netcommon
