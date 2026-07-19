#include "net_stack.h"

#include <chrono>
#include <cstring>
#include <vector>

namespace userspace_net {

NetStack::NetStack(afxdp::Config config) : socket_(std::move(config)) {}

NetStack::~NetStack() { stop(); }

bool NetStack::start(RecvCallback on_recv) {
  if (!socket_.open()) return false;
  running_ = true;
  pollThread_ = std::thread(&NetStack::pollLoop, this, std::move(on_recv));
  return true;
}

void NetStack::stop() {
  running_ = false;
  if (pollThread_.joinable()) pollThread_.join();
  socket_.close();
}

void NetStack::pollLoop(RecvCallback on_recv) {
  // Busy-poll, not epoll — see file-level comment. This burns a full
  // core; that's the accepted tradeoff for the latency this design is
  // chasing (same tradeoff cpu_engine's busy-poll-vs-OS-wait step,
  // Phase 2 step 13, measured for the SPSC ring buffer case).
  std::pair<uint32_t, uint32_t> received[64];
  while (running_.load(std::memory_order_relaxed)) {
    size_t n = socket_.receive(received, 64);
    for (size_t i = 0; i < n; ++i) {
      auto [frameIdx, len] = received[i];
      const uint8_t *frame = socket_.umemFrame(frameIdx);
      if (len >= sizeof(FrameHeader)) {
        FrameHeader header;
        std::memcpy(&header, frame, sizeof(header));
        if (header.magic == kMagic && sizeof(FrameHeader) + header.payload_len <= len)
          on_recv(frame + sizeof(FrameHeader), header.payload_len);
      }
      socket_.recycle(frameIdx);
    }
  }
}

bool NetStack::send(const void *payload, uint32_t len, int timeout_us) {
  std::vector<uint8_t> frame(sizeof(FrameHeader) + len);
  FrameHeader header{kMagic, len};
  std::memcpy(frame.data(), &header, sizeof(header));
  std::memcpy(frame.data() + sizeof(header), payload, len);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
  while (!socket_.transmit(frame.data(), frame.size())) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

} // namespace userspace_net
