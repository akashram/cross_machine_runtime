//===- af_xdp_socket.h - AF_XDP UMEM + TX/RX ring wrapper ----------------===//
//
// Requires Linux + libbpf (xsk.h) + a NIC/driver with native or generic
// XDP support. AF_XDP lets userspace poll TX/RX descriptor rings that
// point directly into a shared UMEM region — packets never cross into
// the kernel's normal socket buffer path. This is the kernel-bypass layer
// step 9 (userspace_net) builds a send/recv pipeline on top of.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct xsk_socket;
struct xsk_umem;
struct xsk_ring_cons;
struct xsk_ring_prod;

namespace afxdp {

struct Config {
  std::string ifname;
  uint32_t queue_id = 0;
  uint32_t num_frames = 4096;
  uint32_t frame_size = 2048; // must be a power of 2, >= MTU
};

// Owns one UMEM (a single mmap'd region, `num_frames * frame_size`
// bytes, sliced into fixed-size frames) plus the four rings AF_XDP
// exposes: fill/completion (RX-side and TX-side frame handoff to/from the
// kernel) and RX/TX (actual packet descriptors). See af_xdp_socket.cpp
// for the setup order — it matters: UMEM must exist before the socket is
// created, and the fill ring must be pre-populated with free frames
// before the socket can receive anything.
class XdpSocket {
public:
  explicit XdpSocket(Config config);
  ~XdpSocket();

  XdpSocket(const XdpSocket &) = delete;
  XdpSocket &operator=(const XdpSocket &) = delete;

  bool open();
  void close();

  // Polls the RX ring for up to `max_frames` received packets, filling
  // `out` with (frame index, length) pairs whose bytes live in the UMEM
  // at `frame_index * frame_size`. Caller must call recycle() once done
  // reading, to return the frame to the fill ring for reuse.
  size_t receive(std::pair<uint32_t, uint32_t> *out, size_t max_frames);
  void recycle(uint32_t frame_index);

  // Copies `len` bytes into a free UMEM frame and submits it on the TX
  // ring. Returns false if no frame was available (fill/completion ring
  // exhausted — caller should drain completions and retry).
  bool transmit(const void *data, size_t len);

  uint8_t *umemFrame(uint32_t frame_index);

private:
  Config config_;
  void *umemArea_ = nullptr;
  xsk_umem *umem_ = nullptr;
  xsk_socket *socket_ = nullptr;
  xsk_ring_prod *fillRing_ = nullptr;
  xsk_ring_cons *completionRing_ = nullptr;
  xsk_ring_cons *rxRing_ = nullptr;
  xsk_ring_prod *txRing_ = nullptr;
};

} // namespace afxdp
