#include "sla_scheduler.h"

#include <algorithm>
#include <cstdio>

using namespace inference_serving;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

bool contains(const std::vector<int> &v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

void test_admits_earliest_deadline_first() {
  SlaScheduler s;
  // id 0: deadline 5.0 (tight); id 1: deadline 20.0; id 2: deadline 10.0
  s.add_request({0, /*arrival=*/0.0, /*budget=*/5.0});
  s.add_request({1, 0.0, 20.0});
  s.add_request({2, 0.0, 10.0});

  auto tick = s.schedule_tick(/*max_batch_size=*/2, /*now_s=*/0.0);
  bool got_0_and_2 = tick.active_seq_ids.size() == 2 && contains(tick.active_seq_ids, 0) &&
                      contains(tick.active_seq_ids, 2) && !contains(tick.active_seq_ids, 1);
  require(got_0_and_2, "EDF admits the two earliest-deadline requests, not arrival order");
}

void test_preempts_for_more_urgent_arrival() {
  SlaScheduler s;
  s.add_request({0, 0.0, 100.0});  // deadline 100, admitted first
  s.add_request({1, 0.0, 100.0});  // deadline 100, admitted first
  auto first = s.schedule_tick(/*max_batch_size=*/2, 0.0);
  require(first.active_seq_ids.size() == 2 && first.preempted_seq_ids.empty(), "fills both slots, no preemption yet");

  s.add_request({2, 1.0, 1.0});  // deadline 2.0 -- far more urgent than 0/1's deadline 100
  auto second = s.schedule_tick(2, 1.0);
  bool preempted_one_of_the_originals = second.preempted_seq_ids.size() == 1 &&
      (second.preempted_seq_ids[0] == 0 || second.preempted_seq_ids[0] == 1);
  bool urgent_now_active = contains(second.active_seq_ids, 2);
  require(preempted_one_of_the_originals && urgent_now_active,
          "preempts a less-urgent active request to admit a newly-arrived urgent one");
  require(s.num_preemptions() == 1, "preemption counter reflects exactly one eviction");
}

void test_sla_violation_counted_once() {
  SlaScheduler s;
  s.add_request({0, /*arrival=*/0.0, /*budget=*/1.0});  // deadline 1.0
  s.schedule_tick(4, /*now_s=*/0.5);   // before deadline: no violation
  s.schedule_tick(4, /*now_s=*/2.0);   // past deadline: 1 violation
  s.schedule_tick(4, /*now_s=*/3.0);   // still pending, past deadline: must NOT double-count
  require(s.num_sla_violations() == 1, "violation counted exactly once per request, not once per tick");
}

void test_finish_removes_from_all_tracking() {
  SlaScheduler s;
  s.add_request({0, 0.0, 1.0});
  s.schedule_tick(4, 0.0);
  s.finish_sequence(0);
  require(s.num_pending() == 0, "finish_sequence removes the request from pending tracking");
  s.schedule_tick(4, 5.0);  // well past its old deadline
  require(s.num_sla_violations() == 0, "a finished request can't retroactively count as an SLA violation");
}

} // namespace

int main() {
  test_admits_earliest_deadline_first();
  test_preempts_for_more_urgent_arrival();
  test_sla_violation_counted_once();
  test_finish_removes_from_all_tracking();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
