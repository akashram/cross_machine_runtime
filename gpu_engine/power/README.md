# NVML Power Monitoring

**Status: STUB — requires CUDA GPU + NVML. Run on any CUDA Linux instance.**

## What this measures
Per-kernel power draw (W) via `nvmlDeviceGetPowerUsage`, temperature during sustained
load, and thermal throttling detection. Integrated into the benchmark harness so every
benchmark reports power alongside latency.

## Implementation notes
- NVML sampling: poll `nvmlDeviceGetPowerUsage` every 10ms in a background thread
- Throttle detection: `nvmlDeviceGetCurrentClocksThrottleReasons` (bitmask)
  - `nvmlClocksThrottleReasonSwThermalSlowdown` = software thermal throttle
  - `nvmlClocksThrottleReasonHwThermalSlowdown` = hardware thermal throttle
- Temperature: `nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp)`
- Energy per inference: integrate power(t) dt over kernel duration
- Watch for sustained throttle on p4d (A100 TDP = 400W, rack PDU limits)

## Results

TODO: run on GPU hardware and fill in this table.

| Kernel | Avg power (W) | Peak power (W) | Temp (°C) | Throttled? |
|--------|--------------|----------------|-----------|------------|
| Idle | TODO | TODO | TODO | TODO |
| elementwise_add | TODO | TODO | TODO | TODO |
| gemm_wmma (sustained) | TODO | TODO | TODO | TODO |
| flash_attn_fwd | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU on Linux (NVML requires root or nvidia-smi group)
- Build preset: cuda (Linux)
- Link: `-lnvidia-ml`
