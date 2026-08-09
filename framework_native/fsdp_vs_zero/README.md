# fsdp_vs_zero

**Status: code-complete AND locally run — `.venv` (`torch==2.2.2`), real
multi-process.**

## What this measures

PLAN.md Phase 19 step 4: a real `torch.distributed.fsdp.
FullyShardedDataParallel` run with `ShardingStrategy.FULL_SHARD`
(shards parameters, gradients, AND optimizer state across ranks — the
structural definition of Rajbhandari et al. 2020's ZeRO stage 3), with a
direct structural comparison against `distributed_training/zero1`/
`zero2`/`zero3`'s hand-written sharding.

## Design

- 4 real separate OS processes (`torch.multiprocessing.spawn`, `gloo`
  backend), same real-multi-process standard as step 3.
- Correctness check mirrors `zero1`/`zero2`/`zero3`'s own methodology
  exactly: the loss curve under sharding must match a no-sharding
  single-process baseline, since sharding changes WHERE state lives, not
  the training math (see those steps' own "Sanity-run output" sections).
- Real, measured evidence of sharding: each rank's LOCAL parameter-tensor
  element count is checked directly (not assumed from the class name)
  against the full model's element count.

## Results (captured 2026-08-09, `torch==2.2.2`, this Mac)

```
  4 REAL separate OS processes (torch.multiprocessing.spawn), FSDP FULL_SHARD (= ZeRO-3 sharding: params+grads+optimizer state)
  full model parameter elements: 144 | rank 0's LOCAL shard: 36 (25.0% of full)

  step   baseline_loss   fsdp_rank0_shard_loss
     0   0.048226      0.048542
    10   0.032134      0.033639
    20   0.028677      0.029927
    30   0.027036      0.027990
    40   0.025833      0.026515
    50   0.024840      0.025264
    59   0.024080      0.024281

  final single-process loss (full dataset) = 0.024080
  final FSDP global loss (all 400 samples, gathered weights) = 0.024002

PASS  FSDP's final global loss matches the single-process baseline's (sharding changes WHERE state lives, not the training math -- same claim zero1/zero2/zero3 verify)
PASS  each rank's LOCAL parameter count is strictly smaller than the full model's -- real sharding measured directly, not assumed from the API name
```

## Findings

**Three real bugs, found by actually running this, not by reading FSDP's
docs — kept visible in the source rather than quietly fixed:**

1. **`torch==2.2.2`'s FSDP assumes CUDA exists unless told otherwise.**
   `FSDP._init_device_handle()` falls back to `torch.device("cuda",
   torch.cuda.current_device())` whenever no `device_id` is passed and
   every parameter is already on `cpu`/`meta` — it never treats "this
   really is a CPU-only run" as a valid outcome on its own. Every worker
   crashed with `AssertionError: Torch not compiled with CUDA enabled`.
   Fixed by passing `device_id=torch.device("cpu")` explicitly to the
   `FSDP()` constructor.
2. **`FSDP.summon_full_params()` is a COLLECTIVE operation**, needing
   every rank to enter it together (the same way `dist.barrier()` does) —
   the first version of this script gated it behind `if rank == 0:`
   ("only rank 0 needs the full params to measure them," reasonable-
   looking but wrong), leaving ranks 1-3 skipping the collective and
   exiting immediately while rank 0 hung forever waiting for peers that
   had already left. Diagnosed by watching process state directly: all 4
   gloo ranks connected (4 hostname-resolution warnings visible), then
   CPU time froze with only rank 0's process still alive. Fixed by having
   every rank call `summon_full_params`, gating only the save/measurement
   step on rank 0.
3. **Using the plain (pre-FSDP-wrap) `model` object outside
   `summon_full_params`'s context reads sharded storage.** FSDP shards a
   module's parameters IN PLACE, so `model` and `fsdp_model` share
   underlying storage — calling `model`'s forward pass after
   `summon_full_params`'s context had already exited hit
   `RuntimeError: setStorage: ... storage size of 0`. Fixed by moving that
   forward pass inside the (already-entered-by-every-rank) context.
- **Real, measured evidence FSDP is actually sharding, not just DDP under
  a different name**: rank 0's local parameter count is exactly `36` out
  of `144` total — exactly `25.0%`, `1/world_size` for 4 ranks, matching
  precisely what ZeRO-3's own design targets (`zero3/README.md`: "peak
  memory during a single step also stays near `1/world_size`").
- **Final losses agree closely** (`0.024080` single-process vs. `0.024002`
  FSDP) — not bit-exact like step 3's DDP comparison, because FSDP's
  FULL_SHARD introduces additional floating-point-order-of-operations
  differences beyond DDP's (parameters themselves are gathered/scattered
  across the training loop, not just gradients), but well within the same
  "sharding doesn't change the training math" claim `zero1`/`zero2`/
  `zero3` all verify for their own hand-written implementations.

## Structural comparison to `zero1`/`zero2`/`zero3`

| | Shards params | Shards grads | Shards optimizer state | FSDP equivalent |
|---|---|---|---|---|
| `zero1` | no | no | yes | none directly (partial) |
| `zero2` | no | yes | yes | `ShardingStrategy.SHARD_GRAD_OP` |
| `zero3` | yes | yes | yes | `ShardingStrategy.FULL_SHARD` (tested here) |

`zero1`/`zero2`/`zero3` each independently implement one point on this
same progression Rajbhandari et al. (2020) define; FSDP's
`ShardingStrategy` enum is PyTorch's first-class API for the identical
progression — this step validates the FULL_SHARD (ZeRO-3) end of it
directly against a real production implementation, with the same
correctness bar this repo's own hand-written versions are held to.

## Hardware notes
CPU only, `gloo` backend. 4 processes on one Mac.
