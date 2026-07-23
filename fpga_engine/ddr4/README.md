# ddr4 — multi-bank DDR4 integration

**Status: code complete — requires Vitis HLS + v++ on AWS F1 to build.**

## What this measures
F1's VU9P card has 4 independent DDR4 banks, ~19.25 GB/s each,
~77 GB/s aggregate. `ddr4_bandwidth.cpp` runs 4 independent streaming
copies concurrently via `DATAFLOW`; `ddr4.cfg`'s `sp=` directives are what
actually pin each copy's `m_axi` bundle to a distinct bank — without them,
v++ is free to place every bundle on bank 0, capping the whole kernel at
one bank's bandwidth regardless of how the HLS side is structured.

The multi-bank strategy this step is meant to validate: independent
DATAFLOW tasks, one per bank, each with its own `m_axi` bundle explicitly
bound via the link config — as opposed to, say, one wide interleaved
access pattern across all 4 banks from a single task (a different,
untested strategy that would need bank-interleaved addressing logic
instead of static per-task binding).

## Results
TODO: build + run on F1.

| Config | Achieved bandwidth (GB/s) | % of per-bank peak (19.25 GB/s) |
|---|----|----|
| Bank 0 alone | TODO | TODO |
| All 4 banks concurrent (this kernel) | TODO | TODO (target: near 77 GB/s aggregate) |

## Hardware notes
- Required: AWS F1, Vitis HLS + v++ 2022.x
- Build: `v++ -c -k ddr4_bandwidth ddr4_bandwidth.cpp -o ddr4_bandwidth.xo`,
  then `v++ --link --config ddr4.cfg ddr4_bandwidth.xo -o ddr4_bandwidth.xclbin`
