//===- af_xdp_socket.cpp - AF_XDP setup, mirroring libbpf's xdpsock ------===//
//
// Setup order (this is the part that's easy to get wrong and silently
// hang instead of erroring): UMEM must be created — and its fill ring
// pre-populated with every frame index — *before* xsk_socket__create,
// because the kernel starts trying to use the fill ring the moment the
// socket exists. Getting this backwards is the classic AF_XDP "RX ring
// never delivers anything" bug.
//
//===----------------------------------------------------------------------===//

#include "af_xdp_socket.h"

#include <bpf/xsk.h>
#include <sys/mman.h>

#include <cstring>
#include <stdexcept>

namespace afxdp {

XdpSocket::XdpSocket(Config config) : config_(std::move(config)) {}

XdpSocket::~XdpSocket() { close(); }

bool XdpSocket::open() {
  size_t umemSize = static_cast<size_t>(config_.num_frames) * config_.frame_size;
  umemArea_ = ::mmap(nullptr, umemSize, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (umemArea_ == MAP_FAILED) {
    // Hugepages often aren't configured — fall back to a normal mapping
    // rather than failing outright; TLB pressure is a perf concern here,
    // not a correctness one.
    umemArea_ = ::mmap(nullptr, umemSize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (umemArea_ == MAP_FAILED) return false;
  }

  xsk_umem_config umemCfg{};
  umemCfg.fill_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
  umemCfg.comp_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
  umemCfg.frame_size = config_.frame_size;
  umemCfg.frame_headroom = XSK_UMEM__DEFAULT_FRAME_HEADROOM;

  // xsk_umem__create fills these ring structs in place — the pointers
  // must already point at real storage before the call, not be created by it.
  fillRing_ = new xsk_ring_prod();
  completionRing_ = new xsk_ring_cons();
  if (xsk_umem__create(&umem_, umemArea_, umemSize, fillRing_, completionRing_, &umemCfg) != 0)
    return false;

  // Pre-populate the fill ring with every frame — this is what makes the
  // kernel able to deliver RX packets into this UMEM at all (see the
  // file-level comment on setup order).
  uint32_t idx;
  uint32_t reserved = xsk_ring_prod__reserve(fillRing_, config_.num_frames, &idx);
  for (uint32_t i = 0; i < reserved; ++i)
    *xsk_ring_prod__fill_addr(fillRing_, idx + i) = static_cast<uint64_t>(i) * config_.frame_size;
  xsk_ring_prod__submit(fillRing_, reserved);

  xsk_socket_config xskCfg{};
  xskCfg.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
  xskCfg.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
  xskCfg.libbpf_flags = 0;
  xskCfg.xdp_flags = 0; // let libxdp pick native vs. generic (driver-dependent)
  xskCfg.bind_flags = 0;

  rxRing_ = new xsk_ring_cons();
  txRing_ = new xsk_ring_prod();
  if (xsk_socket__create(&socket_, config_.ifname.c_str(), config_.queue_id, umem_, rxRing_,
                          txRing_, &xskCfg) != 0)
    return false;

  return true;
}

void XdpSocket::close() {
  if (socket_) { xsk_socket__delete(socket_); socket_ = nullptr; }
  if (umem_) { xsk_umem__delete(umem_); umem_ = nullptr; }
  if (umemArea_) {
    ::munmap(umemArea_, static_cast<size_t>(config_.num_frames) * config_.frame_size);
    umemArea_ = nullptr;
  }
  delete fillRing_; fillRing_ = nullptr;
  delete completionRing_; completionRing_ = nullptr;
  delete rxRing_; rxRing_ = nullptr;
  delete txRing_; txRing_ = nullptr;
}

size_t XdpSocket::receive(std::pair<uint32_t, uint32_t> *out, size_t max_frames) {
  uint32_t idx;
  uint32_t available = xsk_ring_cons__peek(rxRing_, static_cast<uint32_t>(max_frames), &idx);
  for (uint32_t i = 0; i < available; ++i) {
    const xdp_desc *desc = xsk_ring_cons__rx_desc(rxRing_, idx + i);
    out[i] = {static_cast<uint32_t>(desc->addr / config_.frame_size), desc->len};
  }
  xsk_ring_cons__release(rxRing_, available);
  return available;
}

void XdpSocket::recycle(uint32_t frame_index) {
  uint32_t idx;
  if (xsk_ring_prod__reserve(fillRing_, 1, &idx) != 1) return; // fill ring full — drop
  *xsk_ring_prod__fill_addr(fillRing_, idx) = static_cast<uint64_t>(frame_index) * config_.frame_size;
  xsk_ring_prod__submit(fillRing_, 1);
}

bool XdpSocket::transmit(const void *data, size_t len) {
  if (len > config_.frame_size) throw std::runtime_error("af_xdp: packet exceeds frame_size");

  // Reclaim any completed TX frames first — this is the only source of
  // free frames for transmit besides the initial fill.
  uint32_t compIdx;
  uint32_t completed = xsk_ring_cons__peek(completionRing_, config_.num_frames, &compIdx);
  static uint32_t nextFreeFrame = 0; // simple bump allocator over reclaimed frames
  xsk_ring_cons__release(completionRing_, completed);

  uint32_t txIdx;
  if (xsk_ring_prod__reserve(txRing_, 1, &txIdx) != 1) return false;

  uint32_t frame = nextFreeFrame++ % config_.num_frames;
  std::memcpy(umemFrame(frame), data, len);

  xdp_desc *desc = xsk_ring_prod__tx_desc(txRing_, txIdx);
  desc->addr = static_cast<uint64_t>(frame) * config_.frame_size;
  desc->len = static_cast<uint32_t>(len);
  xsk_ring_prod__submit(txRing_, 1);
  return true;
}

uint8_t *XdpSocket::umemFrame(uint32_t frame_index) {
  return static_cast<uint8_t *>(umemArea_) + static_cast<size_t>(frame_index) * config_.frame_size;
}

} // namespace afxdp
