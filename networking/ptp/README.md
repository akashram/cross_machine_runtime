# PTP Clock Synchronization (IEEE 1588)

**Status: STUB — requires 2+ Linux nodes with PTP-capable NICs.**

## What this measures
Precision Time Protocol synchronization accuracy between nodes.
Compare ptp4l + phc2sys vs NTP baseline. Target: < 1µs offset.

## Results
TODO: run on multi-node setup.

| Protocol | Offset mean (ns) | Offset stddev (ns) | Max error (ns) |
|----------|-----------------|-------------------|----------------|
| NTP | TODO | TODO | TODO |
| PTP (hardware timestamping) | TODO | TODO | TODO |

## Hardware notes
- Required: NICs with hardware timestamping (ENA on AWS supports this)
- Software: linuxptp package (ptp4l, phc2sys)
