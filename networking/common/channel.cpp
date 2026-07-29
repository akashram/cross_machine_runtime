//===- channel.cpp - TcpChannel implementation ----------------------------===//

#include "channel.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

namespace netcommon {
namespace {

// A real, pre-existing gap this project's socket layer had, surfaced
// (not caused) by RaftNode::stop()'s shutdownPeer() fix: writing to a
// socket whose peer has reset or half-closed the connection raises
// SIGPIPE by default, and SIGPIPE's default disposition is to terminate
// the whole process -- silently, with no exception, no unwind, nothing
// for send()'s caller to even catch. This affected every Channel user
// in Phase 5, not just Raft; it just never manifested before because
// nothing previously did a live socket shutdown while a peer kept
// sending. Ignoring SIGPIPE process-wide is the standard, portable fix
// for any POSIX networked application (works identically on macOS and
// Linux, unlike SO_NOSIGPIPE which is BSD/macOS-only) -- it turns a
// broken-pipe write into an ordinary send() failure (errno == EPIPE),
// which sendAll below already correctly turns into a catchable
// std::runtime_error.
struct IgnoreSigpipe {
  IgnoreSigpipe() { std::signal(SIGPIPE, SIG_IGN); }
};
const IgnoreSigpipe kIgnoreSigpipeOnce;

void sendAll(int fd, const void *data, size_t len) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, p + sent, len - sent, 0);
    if (n <= 0) throw std::runtime_error("Channel: send failed: " + std::string(strerror(errno)));
    sent += static_cast<size_t>(n);
  }
}

void recvAll(int fd, void *data, size_t len) {
  uint8_t *p = static_cast<uint8_t *>(data);
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, p + got, len - got, 0);
    if (n <= 0) throw std::runtime_error("Channel: recv failed/closed: " + std::string(strerror(errno)));
    got += static_cast<size_t>(n);
  }
}

int makeListenSocket(uint16_t port, const std::string &host) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("Channel: socket() failed");
  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
    throw std::runtime_error("Channel: bind() failed on port " + std::to_string(port));
  if (::listen(fd, /*backlog=*/64) != 0)
    throw std::runtime_error("Channel: listen() failed");
  return fd;
}

int connectWithRetry(uint16_t port, const std::string &host) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

  // The peer's listen() may not have run yet — this constructor doubles as
  // the mesh barrier (see channel.h), so a short retry loop here is the
  // whole synchronization mechanism, not a workaround for a bug.
  for (int attempt = 0; attempt < 2000; ++attempt) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("Channel: socket() failed");
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      return fd;
    }
    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  throw std::runtime_error("Channel: could not connect to port " + std::to_string(port));
}

} // namespace

TcpChannel::TcpChannel(int rank, int world_size, uint16_t base_port, const std::string &host)
    : rank_(rank), world_size_(world_size), fds_(world_size, -1) {
  int listenFd = makeListenSocket(static_cast<uint16_t>(base_port + rank), host);

  // Accept from every higher rank (they dial us); connect out to every
  // lower rank (we dial them). This ordering — accept-then-connect, with
  // higher ranks always initiating — is what avoids every pair racing to
  // both `connect()` each other at once.
  int numAccepts = world_size - 1 - rank;
  for (int i = 0; i < numAccepts; ++i) {
    sockaddr_in peerAddr{};
    socklen_t peerLen = sizeof(peerAddr);
    int fd = ::accept(listenFd, reinterpret_cast<sockaddr *>(&peerAddr), &peerLen);
    if (fd < 0) throw std::runtime_error("Channel: accept() failed");
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int32_t peerRankNet = 0;
    recvAll(fd, &peerRankNet, sizeof(peerRankNet));
    int peerRank = ntohl(peerRankNet);
    fds_[peerRank] = fd;
  }
  ::close(listenFd);

  for (int j = 0; j < rank; ++j) {
    int fd = connectWithRetry(static_cast<uint16_t>(base_port + j), host);
    int32_t myRankNet = htonl(rank);
    sendAll(fd, &myRankNet, sizeof(myRankNet));
    fds_[j] = fd;
  }
}

TcpChannel::~TcpChannel() {
  for (int fd : fds_)
    if (fd >= 0) ::close(fd);
}

void TcpChannel::send(int dst_rank, const void *data, size_t len) {
  sendAll(fds_.at(dst_rank), data, len);
}

void TcpChannel::recv(int src_rank, void *data, size_t len) {
  recvAll(fds_.at(src_rank), data, len);
}

void TcpChannel::shutdownPeer(int peer) {
  int fd = fds_.at(peer);
  if (fd >= 0) ::shutdown(fd, SHUT_RD);
  // SHUT_RD only, deliberately: this is a local-only operation on our
  // own read side, guaranteed (POSIX) to unblock a concurrent thread
  // blocked in recv() on this fd by making it return 0 -- without
  // requiring the peer's cooperation, and without disturbing our own
  // send() path (a concurrently in-flight send to this peer, e.g. from
  // the ticker thread mid-heartbeat, must still complete normally; see
  // raft.cpp's stop()).
}

std::vector<std::unique_ptr<Channel>> make_tcp_loopback_mesh(int world_size, uint16_t base_port) {
  std::vector<std::future<std::unique_ptr<Channel>>> futures;
  futures.reserve(world_size);
  for (int r = 0; r < world_size; ++r) {
    futures.push_back(std::async(std::launch::async, [r, world_size, base_port] {
      return std::unique_ptr<Channel>(std::make_unique<TcpChannel>(r, world_size, base_port));
    }));
  }
  std::vector<std::unique_ptr<Channel>> channels;
  channels.reserve(world_size);
  for (auto &f : futures) channels.push_back(f.get());
  return channels;
}

} // namespace netcommon
