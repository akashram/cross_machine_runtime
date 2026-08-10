# Backpressure + Load Shedding

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Token bucket rate limiting, explicit backpressure signals between nodes,
graceful degradation under overload.

## Design
Two composable primitives (`backpressure.cpp`): `TokenBucket` is
self-contained local rate limiting (no network dependency at all —
capacity + continuous refill, non-blocking `tryAcquire`). `BackpressureSender`
/ `BackpressureReceiver` are **credit-based flow control** over
`networking/common::Channel`: the sender starts with `windowSize` credits,
consumes one per send via `acquireCredit()` (blocking if none are left),
and the receiver grants exactly one credit back per data unit it actually
*finishes processing* via `grantCredit()` — not per unit merely received.
Control and data use opposite directions of the same full-duplex socket —
no framing or tagging needed for the 1-byte credit grant, since the
receiver never sends data and the sender never sends control.

**Two real findings, not just design notes** — the second one replaced
the original design after it broke under real conditions:

1. An early version had the sender flood as fast as `send()` would allow,
   honoring only a receiver-driven PAUSE signal sent when queue depth
   crossed a watermark. Queue depth blew past 50 against a 20-item high
   watermark. Real reason: TCP buffers far more than a few dozen 1-byte
   messages across its send buffer, the network, and the receive buffer,
   so a PAUSE — which only takes effect once the receiver notices and the
   signal propagates — cannot bound how much is already in flight by the
   time the sender sees it. Fixed at the time by composing a `TokenBucket`
   sender-side rate limit with PAUSE/RESUME.

2. **That fix was still not a hard bound, and it eventually showed up as a
   real flaky test** (2026-08-10, caught by a full-suite `ctest` run under
   concurrent system load: max depth hit 26 against a `high_watermark + 5`
   = 25 assertion). Root cause: PAUSE/RESUME is a *threshold-crossing
   signal* — its effectiveness depends on how fast it propagates and gets
   scheduled, which has no fixed ceiling under real OS scheduling, so
   `overshoot ≈ (send_rate − drain_rate) × signal_delay` is unbounded in
   principle no matter how generous the slack constant is. Confirmed by
   reproducing it standalone as genuine timing-dependent flakiness (5
   consecutive isolated reruns all passed comfortably), not a logic bug in
   the mechanism itself. **Rather than widen the slack constant**, the
   design was replaced with credit-based flow control (this file, as
   described above) — the same technique TCP's receive window, HTTP/2's
   flow-control window, and gRPC's per-stream window use for exactly this
   reason. `acquireCredit()` makes it *structurally impossible* to send a
   `(windowSize+1)`-th unit without a grant, so `max_in_flight <=
   windowSize` holds by construction, independent of scheduling delay —
   not an empirically-tuned probability. Verified directly: 10 standalone
   runs and 8 runs under full 4-core CPU saturation (`yes > /dev/null` on
   every core) all hit exactly `max queue depth = 20 = window`, never
   above, matching the bound exactly rather than approaching it from
   below by luck.

3. **Writing the credit-based redesign surfaced a third, unrelated real
   bug under TSan** (2026-08-10): `BackpressureSender::stop()` detached
   (rather than joined) its listener thread, reasoning it was "just
   blocked in `recv()` forever, so harmless to abandon." That is the
   *exact* bug `raft.cpp`'s `RaftNode::stop()` already hit as a real
   SIGSEGV and fixed (see `raft.cpp:143-179` and
   `project_raft_test_segfault_resolved` in project memory) —
   `backpressure.cpp` had copied the old, pre-fix pattern and its comment
   even cited "see raft.cpp's stop() for why detach, not join," a
   citation that had gone stale once raft.cpp stopped detaching. TSan
   caught it directly: `channels` (the test's `vector<unique_ptr<Channel>>`)
   destructing while the detached listener thread was still inside a
   blocking `recv()` referencing it — a real data race /
   destroy-out-from-under-a-live-thread bug, not a false positive. Fixed
   the same way raft.cpp was: `stop()` now calls `channel_.shutdownPeer(peer_)`
   (forcibly unblocking the in-flight `recv()` with no cooperation needed
   from the peer) and then `join()`s, not `detach()`s — so by the time
   `stop()` returns, the listener thread has genuinely exited and
   `channel_` can be safely destroyed. Verified: 5/5 clean runs under
   TSan afterward (previously reproduced the race on the first run).

## Sanity-run output (Mac, loopback, 2026-08-10)

`backpressure_test`: `TokenBucket` capacity/refill check, then a 500-packet
flood from rank 0 to rank 1, credit-limited to a 20-unit window, with
rank 1 processing at a fixed slow rate (simulating a receiver that can't
keep up) and granting credit only as it actually drains:

```
TokenBucket: drained exactly capacity=5 up front (5), refilled after wait (1): PASS
Backpressure: max queue depth seen = 20 (window=20), sender blocked 480 times: PASS
PASS
```

Run 10 consecutive times (standalone) and 8 more times under full 4-core
CPU saturation — `max queue depth seen` was exactly 20 in every single
run, never once above the window, and the sender blocked waiting for
credit hundreds of times per run (proving backpressure is genuinely
engaging, not the sender coincidentally being slow enough on its own).

## Results
This step is a correctness/behavior primitive, not a hardware-dependent
performance benchmark — no Results table. Real network latency (vs.
loopback) would change the *timing* of credit-grant round trips, not the
`max_in_flight <= windowSize` guarantee, which holds regardless.

## Hardware notes
- Builds and runs anywhere (validated on Mac: 10 standalone runs, 8 runs
  under full 4-core CPU saturation, 5 runs under TSan, all clean); no
  hardware dependency at all.
