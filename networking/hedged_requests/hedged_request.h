//===- hedged_request.h - Duplicate slow requests to cut tail latency ----===//
//
// Portable — no network dependency; `backends` are arbitrary blocking
// callables, so this composes with networking/common::Channel calls (or
// anything else) without depending on Channel directly.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace hedging {

struct HedgedResult {
  std::string response;
  int winningBackend;
  bool wasHedged; // true if the primary missed hedgeDelay and a backup was launched
};

// Calls backends[0]; if it hasn't returned within `hedgeDelay`, launches
// backends[1] (and so on, if given more than 2) as well, and returns
// whichever completes first. Every backend races into one shared result
// slot (first writer wins); losing backends are NOT cancelled — a
// blocking callable has no cancellation hook here — they keep running
// in the background and their result is discarded. A real deployment
// would want the backend call itself to accept a deadline/cancellation
// token so a losing hedge doesn't waste backend capacity indefinitely;
// out of scope for this primitive, which is about the client-side race,
// not backend-side cancellation.
HedgedResult hedgedCall(const std::vector<std::function<std::string()>> &backends,
                         std::chrono::milliseconds hedgeDelay);

} // namespace hedging
