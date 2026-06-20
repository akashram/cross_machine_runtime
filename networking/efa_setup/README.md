# EFA Setup + Baseline Latency

**Status: STUB — requires 2× p4d.24xlarge in EFA placement group.**

## What this measures
AWS EFA installer validation, fi_info device enumeration, fi_pingpong latency
and bandwidth baseline against TCP socket baseline.

## Results
TODO: run on EFA hardware.

| Metric | EFA | TCP baseline |
|--------|-----|-------------|
| Latency p50 (µs) | TODO | TODO |
| Latency p99 (µs) | TODO | TODO |
| Bandwidth (GB/s) | TODO | TODO |

## Hardware notes
- Required: 2× p4d.24xlarge, same VPC placement group, EFA-enabled
- Install: aws-efa-installer from AWS documentation
