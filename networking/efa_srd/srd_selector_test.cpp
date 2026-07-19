// srd_selector_test.cpp — sanity-checks select_transport() against the
// scenarios this step's README argues about, without needing an EFA NIC.

#include "srd_selector.h"

#include <cstdio>

namespace {
int failures = 0;

void expect(const char *name, EfaTransport got, EfaTransport want) {
  bool ok = got == want;
  std::printf("%-45s %-3s (got %s, want %s)\n", name, ok ? "OK" : "FAIL",
              got == EfaTransport::SRD ? "SRD" : "RC",
              want == EfaTransport::SRD ? "SRD" : "RC");
  if (!ok) ++failures;
}
} // namespace

int main() {
  expect("ring all-reduce (2 peers, ordered)", select_transport({2, 4 << 20, true}), EfaTransport::RC);
  expect("raw two-sided stream (2 peers, ordered)", select_transport({2, 64, true}), EfaTransport::RC);
  expect("MoE all-to-all dispatch (64 peers, sequenced)", select_transport({64, 4096, false}), EfaTransport::SRD);
  expect("gRPC control plane fan-out (16 peers)", select_transport({16, 256, false}), EfaTransport::SRD);
  expect("small ordered cluster below threshold (4 peers, ordered)", select_transport({4, 1024, true}), EfaTransport::RC);

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
