# Ring All-Reduce

**Status: STUB — requires 2+ Linux nodes with network.**

## What this measures
Ring all-reduce implemented from scratch. Bandwidth efficiency vs theoretical
maximum (2*(N-1)/N * message_size / bandwidth). Compare against NCCL baseline.

## Results
TODO: run on multi-node setup.

| Nodes | Message size | Our bandwidth (GB/s) | NCCL (GB/s) | % of theoretical |
|-------|-------------|---------------------|-------------|-----------------|
| 2 | 1 MB | TODO | TODO | TODO |
| 4 | 256 MB | TODO | TODO | TODO |
| 8 | 1 GB | TODO | TODO | TODO |

## Hardware notes
- Required: 2+ Linux nodes with high-bandwidth interconnect
