// tcp_baseline_bench.cpp — the actual "TCP baseline" column in
// efa_setup/README.md and this step's README: a real ping-pong RTT +
// bandwidth benchmark at the same message sizes fi_pingpong uses
// (64B/4KB/1MB), so the EFA-vs-TCP comparison is apples-to-apples instead
// of two numbers from different tools/methodologies. Standalone (doesn't
// reuse networking/common::Channel) because a genuine 2-node deployment
// needs asymmetric bind/connect addresses Channel's loopback-oriented
// design doesn't model — see networking/common/README.md.
//
// Usage:
//   loopback           — server+client in one process over 127.0.0.1 (what
//                         actually runs today; no EFA hardware needed)
//   server [port]      — real 2-node deployment, passive side
//   client <ip> [port] — real 2-node deployment, active side

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint16_t kDefaultPort = 45678;
constexpr int kIterations = 200;
constexpr size_t kSizes[] = {64, 4096, 1048576};

void sendAll(int fd, const void *data, size_t len) {
  const char *p = static_cast<const char *>(data);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, p + sent, len - sent, 0);
    if (n <= 0) throw std::runtime_error("send failed");
    sent += static_cast<size_t>(n);
  }
}

void recvAll(int fd, void *data, size_t len) {
  char *p = static_cast<char *>(data);
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, p + got, len - got, 0);
    if (n <= 0) throw std::runtime_error("recv failed/closed");
    got += static_cast<size_t>(n);
  }
}

// Echoes back every fixed-size message it receives, forever (one client).
void runServer(uint16_t port) {
  int listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  ::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  ::bind(listenFd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  ::listen(listenFd, 1);

  int fd = ::accept(listenFd, nullptr, nullptr);
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  std::vector<char> buf(*std::max_element(std::begin(kSizes), std::end(kSizes)));
  for (size_t size : kSizes) {
    for (int i = 0; i < kIterations; ++i) {
      recvAll(fd, buf.data(), size);
      sendAll(fd, buf.data(), size);
    }
  }
  ::close(fd);
  ::close(listenFd);
}

struct Stats {
  double p50_us, p99_us, bandwidth_gbs;
};

Stats runClientForSize(int fd, size_t size) {
  std::vector<char> buf(size, 'x');
  std::vector<double> rttUs;
  rttUs.reserve(kIterations);

  auto benchStart = Clock::now();
  for (int i = 0; i < kIterations; ++i) {
    auto start = Clock::now();
    sendAll(fd, buf.data(), size);
    recvAll(fd, buf.data(), size);
    rttUs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
  }
  double totalSec = std::chrono::duration<double>(Clock::now() - benchStart).count();

  std::sort(rttUs.begin(), rttUs.end());
  double p50 = rttUs[rttUs.size() / 2];
  double p99 = rttUs[static_cast<size_t>(rttUs.size() * 0.99)];
  double gb = (2.0 * size * kIterations) / 1.0e9; // round trip = 2x bytes on the wire
  return {p50, p99, gb / totalSec};
}

void runClient(const char *serverIp, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, serverIp, &addr.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    std::perror("connect");
    return;
  }
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  std::printf("%-12s %12s %12s %14s\n", "size", "p50 (us)", "p99 (us)", "bandwidth (GB/s)");
  for (size_t size : kSizes) {
    Stats s = runClientForSize(fd, size);
    std::printf("%-12zu %12.2f %12.2f %14.3f\n", size, s.p50_us, s.p99_us, s.bandwidth_gbs);
  }
  ::close(fd);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <loopback|server [port]|client <ip> [port]>\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];

  if (mode == "loopback") {
    std::thread server(runServer, kDefaultPort);
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // let accept() start
    runClient("127.0.0.1", kDefaultPort);
    server.join(); // server loop exits after kIterations per size, same as client's count
    return 0;
  }
  if (mode == "server") {
    uint16_t port = argc > 2 ? static_cast<uint16_t>(std::stoi(argv[2])) : kDefaultPort;
    runServer(port);
    return 0;
  }
  if (mode == "client") {
    if (argc < 3) { std::fprintf(stderr, "usage: %s client <ip> [port]\n", argv[0]); return 2; }
    uint16_t port = argc > 3 ? static_cast<uint16_t>(std::stoi(argv[3])) : kDefaultPort;
    runClient(argv[2], port);
    return 0;
  }
  std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
  return 2;
}
