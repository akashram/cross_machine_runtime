# xadc — die temperature and voltage rail monitoring

**Status: XRT sensor read is code complete and unrun (F1 required); the
JSON parsing/flagging logic is measured locally.**

## What this measures
Split the same way as `clock_gating/` and `power_ci/`, since only one half
needs an F1 instance:

1. **Host-side sensor read** (`xadc_sensors.cpp`): on AWS F1 the physical
   XADC/SYSMON lives inside the static shell, not the user's programmable
   region, so there's no AXI-Lite path to it from a kernel — XRT reads it
   on the host's behalf and exposes the result via
   `xrt::device::get_info<xrt::info::device::thermal|electrical>()`, the
   same sensors `xbutil examine -r thermal` / `-r electrical` prints. Real
   XRT native C++ API usage (same pattern as `dma/dma_orchestration.cpp`),
   not a stub — dumps both JSON reports to disk.
2. **JSON parsing + flagging** (`parse_xadc_json.py`): extracts die
   temperature and rail voltages from the two JSON reports, flags any rail
   more than 5% off its UltraScale+ VU9P nominal voltage. Plain Python, no
   XRT dependency — its self-test (below) is real, passing output today.

Deliberately out of scope here (left to step 25, `thermal_router/`):
deciding a workload-allocation fraction from the temperature reading, and
measuring the router's response latency to a temperature change. This step
only gets a trustworthy `die_temp_c` number into a program; step 25 owns
the policy and the closed loop.

## Schema caveat
`xadc_sensors.cpp`'s query kinds and `parse_xadc_json.py`'s field names
(`thermals[].description`/`temp_C`, `voltages.<rail>.volts`) follow XRT's
documented sensor report schema — the same shape `xbutil examine -f json`
renders from — not a captured report, since no report has been generated
yet. The one thing the parser depends on that's worth flagging explicitly:
it assumes the FPGA die sensor's `description` field is literally `"FPGA"`
among the board's other PCB sensors, and raises rather than silently
returning a board temperature if that assumption is wrong — see
`parse_thermal()`'s `ValueError`. Confirm the real description string
against a captured report before trusting this on F1.

## Results
**Self-test** (measured locally, `python3 parse_xadc_json.py --self-test`,
synthetic thermal/electrical reports shaped like the documented schema,
one rail — `vccaux` — deliberately pushed 8% off nominal so the flagging
logic has something real to catch):

```
parse_xadc_json._self_test: OK
  die_temp_c = 61.0
  rail_flags = ['vccaux: 1.944V is 8.0% off nominal 1.800V (tolerance 5%)']
```

That the die temperature parses out correctly and exactly one rail gets
flagged (the one deliberately pushed off nominal, not `vccint` or
`vccbram`) is by construction — it only proves the parsing/flagging logic
is correct, not that it matches XRT's real output shape.

**Hardware** — TODO: run `xadc_sensors` against a live F1 card, then
`parse_xadc_json.py thermal.json electrical.json`, confirm the `"FPGA"`
description assumption holds, and record real idle/loaded die temperature
and rail readings:

| Condition | Die temp (°C) | vccint (V) | vccaux (V) | vccbram (V) | Rail flags |
|---|---|---|---|---|---|
| Idle (post-bitstream, no kernel running) | TODO | TODO | TODO | TODO | TODO |
| Loaded (ml_kernel running continuously) | TODO | TODO | TODO | TODO | TODO |

## Files
- `xadc_sensors.cpp` — hardware-gated, real XRT API usage; unrun. Writes
  `thermal.json` / `electrical.json` to the given output directory (`.` by
  default).
- `parse_xadc_json.py` — portable, no XRT dependency; run it directly.
  `--self-test` needs no input files.

## Hardware notes
- Required: AWS F1, XRT installed, card enumerated (`xbutil examine`
  works — see `f1_setup/`)
- Build: `g++ -std=c++17 xadc_sensors.cpp -lxrt_coreutil -o xadc_sensors`
  (path to `xrt.h`/`libxrt_coreutil` comes from sourcing
  `/opt/xilinx/xrt/setup.sh` on the F1 AMI)
- Run: `./xadc_sensors . && python3 parse_xadc_json.py thermal.json electrical.json`
