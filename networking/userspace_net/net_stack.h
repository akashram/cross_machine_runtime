//===- net_stack.h - Minimal send/recv pipeline over af_xdp --------------===//
//
// Builds a message-oriented send/recv API on top of af_xdp's XdpSocket
// (step 8): raw Ethernet frames in/out become a fixed-size-header-plus-
// payload protocol, busy-polled rather than epoll-driven since the whole
// point of AF_XDP is avoiding the syscall/interrupt path — a blocking
// poll() call here would reintroduce exactly the latency this step is
// trying to measure the absence of.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "af_xdp_socket.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace userspace_net {

// Minimal frame header: a 4-byte magic (distinguishes our traffic from
// other protocols sharing the NIC in generic/SKB mode) + a 4-byte payload
// length. Real production framing would need addressing; this is scoped
// to the point-to-point benchmark this step's README measures.
struct FrameHeader {
  uint32_t magic;
  uint32_t payload_len;
};
inline constexpr uint32_t kMagic = 0x584450; // "XDP" in hex-ish

using RecvCallback = std::function<void(const uint8_t *payload, uint32_t len)>;

class NetStack {
public:
  explicit NetStack(afxdp::Config config);
  ~NetStack();

  bool start(RecvCallback on_recv);
  void stop();

  // Blocking send: copies `len` bytes into a UMEM frame (via
  // XdpSocket::transmit) with the header prepended. Retries on transient
  // TX-ring-full until `timeout_us` elapses.
  bool send(const void *payload, uint32_t len, int timeout_us = 10000);

private:
  void pollLoop(RecvCallback on_recv);

  afxdp::XdpSocket socket_;
  std::thread pollThread_;
  std::atomic<bool> running_{false};
};

} // namespace userspace_net
