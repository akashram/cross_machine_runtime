//===- ptp_sync.cpp - PHC offset measurement via PTP_SYS_OFFSET ---------===//
//
// Same technique `testptp` (linuxptp's own test tool) and `phc2sys` use:
// PTP_SYS_OFFSET brackets each PHC read with a CLOCK_REALTIME read
// immediately before and after, inside the kernel driver — bracketing
// like this (rather than two userspace clock_gettime() calls around a
// PHC ioctl) is what keeps scheduling jitter from swamping a signal that
// should be sub-microsecond.
//
//===----------------------------------------------------------------------===//

#include "ptp_sync.h"

#include <fcntl.h>
#include <linux/ptp_clock.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int64_t timespecToNs(const ptp_clock_time &ts) {
  return static_cast<int64_t>(ts.sec) * 1'000'000'000LL + ts.nsec;
}

} // namespace

int64_t measure_offset_ns(const char *ptp_device) {
  int fd = ::open(ptp_device, O_RDONLY);
  if (fd < 0) throw std::runtime_error(std::string("ptp_sync: could not open ") + ptp_device);

  ptp_sys_offset request{};
  request.n_samples = 5; // linuxptp's default; more samples = better outlier rejection
  if (::ioctl(fd, PTP_SYS_OFFSET, &request) != 0) {
    ::close(fd);
    throw std::runtime_error("ptp_sync: PTP_SYS_OFFSET ioctl failed — is this a PTP-capable NIC?");
  }
  ::close(fd);

  // ts[2*i] = sys time before PHC read i, ts[2*i+1] = PHC time i,
  // ts[2*i+2] = sys time after. Pick the sample with the tightest
  // before/after bracket (least scheduling jitter) and report PHC time
  // minus the bracket's midpoint.
  int64_t bestOffset = 0;
  int64_t bestBracket = INT64_MAX;
  for (unsigned i = 0; i < request.n_samples; ++i) {
    int64_t before = timespecToNs(request.ts[2 * i]);
    int64_t phc = timespecToNs(request.ts[2 * i + 1]);
    int64_t after = timespecToNs(request.ts[2 * i + 2]);
    int64_t bracket = after - before;
    int64_t midpoint = before + bracket / 2;
    int64_t offset = phc - midpoint;
    if (bracket < bestBracket) { bestBracket = bracket; bestOffset = offset; }
  }
  return bestOffset;
}

ClockSyncStats measure_accuracy(int n_samples, const char *ptp_device) {
  std::vector<double> samples;
  samples.reserve(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    samples.push_back(static_cast<double>(measure_offset_ns(ptp_device)));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  double mean = 0;
  for (double s : samples) mean += s;
  mean /= samples.size();

  double variance = 0;
  double maxAbs = 0;
  for (double s : samples) {
    variance += (s - mean) * (s - mean);
    maxAbs = std::max(maxAbs, std::abs(s));
  }
  variance /= samples.size();

  return {mean, std::sqrt(variance), maxAbs};
}
