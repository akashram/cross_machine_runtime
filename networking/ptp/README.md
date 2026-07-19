# PTP Clock Synchronization (IEEE 1588)

**Status: code-complete, not yet built — requires 2+ Linux nodes with PTP-capable NICs.**

## What this measures
Precision Time Protocol synchronization accuracy between nodes.
Compare ptp4l + phc2sys vs NTP baseline. Target: < 1µs offset.

## Design
`measure_offset_ns()` (`ptp_sync.cpp`) uses `PTP_SYS_OFFSET` — the same
ioctl linuxptp's own `testptp` tool and `phc2sys` use — which brackets
each PHC (PTP Hardware Clock) read with a `CLOCK_REALTIME` read
immediately before and after, *inside the kernel driver*. That bracketing
is what makes a sub-microsecond measurement possible at all: two
userspace `clock_gettime()` calls around a separate ioctl would have
scheduling jitter comparable to or larger than the offset being measured.
Of the `n_samples` bracketed reads the ioctl returns, the one with the
tightest before/after gap (least jitter) is reported. `measure_accuracy()`
repeats this every 1ms for N samples and computes mean/stddev/max — what
ptp4l's discipline loop leaves as residual error, not what ptp4l itself
reports (this is an independent measurement of the same quantity).

## Results
TODO: run on multi-node setup.

| Protocol | Offset mean (ns) | Offset stddev (ns) | Max error (ns) |
|----------|-----------------|-------------------|----------------|
| NTP | TODO | TODO | TODO |
| PTP (hardware timestamping) | TODO | TODO | TODO |

## Hardware notes
- Required: NICs with hardware timestamping (ENA on AWS supports this)
- Software: linuxptp package (ptp4l, phc2sys)
