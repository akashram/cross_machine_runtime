#include "hedged_request.h"

#include <atomic>
#include <future>
#include <memory>
#include <thread>

namespace hedging {

HedgedResult hedgedCall(const std::vector<std::function<std::string()>> &backends,
                         std::chrono::milliseconds hedgeDelay) {
  auto promisePtr = std::make_shared<std::promise<std::pair<int, std::string>>>();
  std::shared_future<std::pair<int, std::string>> sharedFuture = promisePtr->get_future().share();
  auto fulfilled = std::make_shared<std::atomic<bool>>(false);

  auto launch = [&](int idx) {
    std::thread([idx, fn = backends[static_cast<size_t>(idx)], promisePtr, fulfilled] {
      std::string result = fn();
      bool expected = false;
      // First backend to finish wins; every later finisher's result is
      // simply discarded (see header — no cancellation of the losers).
      if (fulfilled->compare_exchange_strong(expected, true)) promisePtr->set_value({idx, result});
    }).detach();
  };

  launch(0);
  bool hedged = false;
  if (sharedFuture.wait_for(hedgeDelay) != std::future_status::ready) {
    hedged = true;
    for (size_t i = 1; i < backends.size(); ++i) launch(static_cast<int>(i));
  }

  auto [winner, response] = sharedFuture.get();
  return {response, winner, hedged};
}

} // namespace hedging
