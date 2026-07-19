# Backpressure + Load Shedding

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Token bucket rate limiting, explicit backpressure signals between nodes,
graceful degradation under overload.

## Design
Two composable primitives (`backpressure.cpp`): `TokenBucket` is
self-contained local rate limiting (no network dependency at all —
capacity + continuous refill, non-blocking `tryAcquire`). `BackpressureSender`
/ `BackpressureReceiver` are explicit signaling over
`networking/common::Channel` — the receiver reports its queue depth after
every change; crossing a high watermark sends a 1-byte PAUSE, dropping to
a low watermark sends RESUME (two thresholds, not one, so depth
oscillating near a single value doesn't toggle PAUSE/RESUME rapidly).
Control and data use opposite directions of the same full-duplex socket —
no framing or tagging needed for the control byte, since the receiver
never sends data and the sender never sends control.

**A real finding, not just a design note**: an early version of the test
had the sender flood as fast as `send()` would allow, honoring only the
receiver's PAUSE signal. Queue depth blew past 50 against a 20-item high
watermark. The reason is real, not a test bug: TCP buffers far more than
a few dozen 1-byte messages across its send buffer, the network, and the
receive buffer, so a receiver-driven PAUSE — which only takes effect after
the receiver *notices* it's falling behind — cannot bound how much is
already in flight by the time the sender sees it. Explicit backpressure
signaling alone doesn't tightly bound anything on a buffered transport;
it has to compose with a bounded sending rate (`TokenBucket`) to actually
work. Fixed by having the sender self-limit with a `TokenBucket` *and*
honor PAUSE/RESUME — exactly the composition `backpressure.h` documents.

## Sanity-run output (Mac, loopback, 2026-07-19)

`backpressure_test`: `TokenBucket` capacity/refill check, then a 500-packet
flood from rank 0 to rank 1 with rank 1 processing at a fixed slow rate
(simulating a receiver that can't keep up) and rank 0 rate-limited to
~4x rank 1's drain rate — fast enough to force real backpressure, bounded
enough not to blow past a small margin over the watermark:

```
TokenBucket: drained exactly capacity=5 up front (5), refilled after wait (1): PASS
Backpressure: max queue depth seen = 22 (high watermark=20), pauses=26, resumes=26: PASS
PASS
```

Run 3 consecutive times, consistent (queue depth 22-23, 26 pause/resume
cycles each time) — PAUSE/RESUME actually engaging repeatedly is what
proves this is backpressure working, not the sender coincidentally being
slow enough on its own.

## Results
This step is a correctness/behavior primitive, not a hardware-dependent
performance benchmark — no Results table. Real network latency (vs.
loopback) would change the *timing* of PAUSE/RESUME propagation delay,
not the mechanism.

## Hardware notes
- Builds and runs anywhere (validated on Mac, 3 repeated runs); no
  hardware dependency at all.
