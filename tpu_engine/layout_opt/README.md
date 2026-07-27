# layout_opt

**Status: portable half done and run locally (`layout_opt_model.cpp`);
hardware half (real MXU utilization before/after on a TPU) TODO — no TPU on
this Mac.**

## What this measures

PLAN.md Phase 8 step 5: tile padding for systolic array alignment, measure
MXU utilization before/after.

## Design

- `layout_opt_model.cpp`: pure geometry, no TPU/JAX dependency. Pads each
  dimension of a matmul up to the MXU's 128x128 tile size and reports the
  *utilization ceiling* — useful FLOPs / padded FLOPs — an upper bound on
  achievable MXU utilization that holds regardless of anything the
  compiler or scheduler does afterward. `tpu_benchmarks/mxu_util_bench.py`'s
  real profiler numbers, once run, should land at or below this ceiling,
  never above it.
- Six shapes: an aligned attention projection, a misaligned `lm_head`
  projection (vocab=50257, the actual source of misalignment in a real
  transformer — d_model and hidden dims are usually chosen as clean
  multiples of 128, vocab size isn't), an aligned vs. off-by-one MLP
  up-projection pair to isolate the "1 past a tile boundary" cliff, and two
  small-batch cases (batch=17, batch=1 decode) since batch dimension
  padding is the real driver of the padding waste this repo's Phase 6
  training work would otherwise account for.
- Build/run directly, no CMake (this is a standalone model, same convention
  as `fpga_engine`'s `*_model.cpp` files):
  `clang++ -O2 -std=c++17 layout_opt_model.cpp -o layout_opt_model && ./layout_opt_model`

## Results (captured 2026-07-27, Apple clang 14, this Mac)

```
shape                                           M        N        K   util ceiling
attn qkv proj, d_model=512 (aligned)          512     1536      512         100.0%
lm_head proj, vocab=50257 (misaligned)       4096    50257      768          99.9%
mlp up-proj, d=4096 (aligned)                4096    16384     4096         100.0%
mlp up-proj, d=4097 (off-by-one)             4097    16384     4097          94.1%
small batch=17 x d=768                         17      768      768          13.3%
batch=1 decode step, d=4096                     1     4096     4096           0.8%
```

## Findings

- The vocab-size misalignment case is nearly free (99.9%) — a 50257 → 50304
  pad wastes only 47 rows against 50304, negligible relative to the whole
  matrix. The real cliff isn't "any misalignment," it's misalignment
  relative to the *smaller* operand dimension: `mlp up-proj, d=4097`
  (padding a dim that's already close to a tile boundary, on a matmul where
  every dim is comparably sized) drops to 94.1%, and small-batch cases drop
  catastrophically (13.3% at batch=17, 0.8% at batch=1) because padding a
  tiny dimension up to a full 128-wide tile wastes almost the entire tile.
- Practical implication for step 9 (mxu_opt) and the training-loop batch
  size choices elsewhere in this repo: batch-dimension alignment to 128
  matters far more than feature-dimension alignment, and single-token
  autoregressive decode (batch=1) is structurally incompatible with high
  MXU utilization on this hardware — a real argument for why serving
  systems (Phase 9) batch decode requests rather than running them one at
  a time, independent of any TPU-specific measurement.

## Hardware notes
- Required for the "before/after on a TPU" half: GCP TPU VM, run
  `tpu_benchmarks/mxu_util_bench.py` at both an aligned and misaligned size
  from the table above and compare the real profiler utilization % against
  this model's ceiling.
