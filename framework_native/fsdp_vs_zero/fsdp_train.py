"""PLAN.md Phase 19 step 4: a real torch.distributed.fsdp.FullyShardedDataParallel
run (FULL_SHARD strategy -- shards parameters, gradients, AND optimizer
state across ranks, the structural definition of Rajbhandari et al.
2020's ZeRO stage 3), with a direct comparison against
`distributed_training/zero1`/`zero2`/`zero3`'s hand-written sharding.

Correctness check mirrors zero1/zero2/zero3's own methodology exactly
(see their READMEs' "Sanity-run output" sections): the loss curve under
sharding must match a no-sharding single-process baseline, since sharding
changes WHERE state lives, not the training math.

Real, separate OS processes via `torch.multiprocessing.spawn` (the
standard, idiomatic launcher PyTorch's own DDP/FSDP tutorials use) with
the `gloo` backend, same real-multi-process standard as step 3's
ddp_gloo. Results are handed back via a temp file (rank 0 writes,
main process reads after join) rather than a `multiprocessing.Queue` --
an earlier version of this script used a raw `Queue` with manually
managed `Process` objects and deadlocked (see README.md's Findings for
the full diagnosis); the file-based handoff avoids that class of problem
entirely and is simpler.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


NUM_SAMPLES = 400
NUM_FEATURES = 8
HIDDEN = 16
NOISE_STD = 0.1
NUM_STEPS = 60
LR = 0.05
WORLD_SIZE = 4


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN, bias=False)
        self.fc2 = nn.Linear(HIDDEN, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def make_dataset(seed: int, model_for_targets: nn.Module):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NUM_SAMPLES, NUM_FEATURES, generator=g)
    with torch.no_grad():
        y_clean = model_for_targets(X)
    noise = torch.randn(NUM_SAMPLES, generator=g) * NOISE_STD
    return X, y_clean + noise


def mse(model, X, y):
    return ((model(X) - y) ** 2).mean()


def train_single_process(target_state, init_state):
    target_model = SmallMLP()
    target_model.load_state_dict(target_state)
    X, y = make_dataset(seed=7, model_for_targets=target_model)

    model = SmallMLP()
    model.load_state_dict(init_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    losses = []
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        loss = mse(model, X, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def fsdp_worker(rank, world_size, target_state, init_state, result_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    target_model = SmallMLP()
    target_model.load_state_dict(target_state)
    X_full, y_full = make_dataset(seed=7, model_for_targets=target_model)
    shard = NUM_SAMPLES // world_size
    start, end = rank * shard, rank * shard + shard
    X, y = X_full[start:end], y_full[start:end]

    model = SmallMLP()
    model.load_state_dict(init_state)
    # FULL_SHARD: parameters, gradients, AND optimizer state all sharded
    # across ranks -- the structural definition of ZeRO stage 3.
    #
    # REAL BUG #1, caught by running this (not by reading FSDP's docs):
    # torch==2.2.2's FSDP._init_device_handle() falls back to
    # `torch.device("cuda", torch.cuda.current_device())` whenever no
    # `device_id` is passed AND every parameter is already on cpu/meta --
    # it never considers "this really is a CPU-only run" a valid outcome
    # on its own. Raises `AssertionError: Torch not compiled with CUDA
    # enabled` inside every worker on a CPU-only build. Fixed by passing
    # `device_id=torch.device("cpu")` explicitly, which sets
    # `determined_device` directly and skips the CUDA-fallback branch.
    fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=torch.device("cpu"))
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=LR)

    losses = []
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        loss = mse(fsdp_model, X, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Each rank only holds a SHARD of the parameters locally (FULL_SHARD's
    # entire memory-saving point) -- the real, measured local
    # parameter-tensor element count per rank, direct evidence FSDP is
    # actually sharding, not just running DDP under a different name.
    local_param_elems = sum(p.numel() for p in fsdp_model.parameters())

    # REAL BUG #2, caught the same way as bug #1 -- by actually running
    # this, not by reading the docs: `FSDP.summon_full_params()` is a
    # COLLECTIVE operation (it all-gathers every rank's shard to
    # reconstruct the full parameter tensors), so it needs EVERY rank to
    # enter the context manager together, the same way `dist.barrier()`
    # or `all_gather()` do. The first version of this script gated it
    # behind `if rank == 0:` (reasonable-looking: "only rank 0 needs the
    # full params to measure them") -- but that left ranks 1-3 skipping
    # the collective entirely and proceeding straight to
    # `destroy_process_group()`, so rank 0 hung forever waiting for
    # collective participants that had already exited. Diagnosed by
    # watching process state directly: all 4 gloo ranks connected
    # (4 hostname-resolution warnings), then CPU time froze with only
    # rank 0's process still alive -- the other 3 had already exited
    # cleanly. Fixed by having EVERY rank call `summon_full_params`,
    # only gating the save/measurement on rank 0.
    # REAL BUG #3, found immediately after fixing #2: `model` (the plain
    # SmallMLP the caller wrapped with FSDP) shares its underlying
    # parameter STORAGE with `fsdp_model` -- FSDP shards a module's
    # parameters IN PLACE, so calling `model`'s forward directly outside
    # `summon_full_params`'s context hits sharded (near-zero-size)
    # storage: `RuntimeError: setStorage: ... storage size of 0`. Fixed
    # by moving the plain-`model` forward pass INSIDE the
    # `summon_full_params` context (which every rank must already be
    # inside for bug #2's fix), where the full, unsharded parameters are
    # temporarily materialized.
    with FSDP.summon_full_params(fsdp_model):
        full_param_elems = sum(p.numel() for p in fsdp_model.parameters())
        with torch.no_grad():
            final_global_loss = mse(model, X_full, y_full).item()

    if rank == 0:
        torch.save(
            {"losses": losses, "final_global_loss": final_global_loss,
             "local_param_elems": local_param_elems, "full_param_elems": full_param_elems},
            result_path,
        )

    dist.destroy_process_group()


def main():
    torch.manual_seed(11)
    target_model = SmallMLP()
    target_state = {k: v.clone() for k, v in target_model.state_dict().items()}
    init_model = SmallMLP()
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}

    baseline_losses = train_single_process(target_state, init_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = os.path.join(tmpdir, "fsdp_result.pt")
        mp.spawn(
            fsdp_worker,
            args=(WORLD_SIZE, target_state, init_state, result_path),
            nprocs=WORLD_SIZE,
            join=True,
        )
        result = torch.load(result_path, weights_only=False)

    fsdp_losses = result["losses"]
    fsdp_final_global_loss = result["final_global_loss"]
    local_param_elems = result["local_param_elems"]
    full_param_elems = result["full_param_elems"]

    print(f"  {WORLD_SIZE} REAL separate OS processes (torch.multiprocessing.spawn), FSDP FULL_SHARD (= ZeRO-3 sharding: params+grads+optimizer state)")
    print(f"  full model parameter elements: {full_param_elems} | rank 0's LOCAL shard: {local_param_elems} ({100.0*local_param_elems/full_param_elems:.1f}% of full)")
    print(f"\n  step   baseline_loss   fsdp_rank0_shard_loss")
    for i in (0, 10, 20, 30, 40, 50, NUM_STEPS - 1):
        print(f"  {i:4d}   {baseline_losses[i]:.6f}      {fsdp_losses[i]:.6f}")

    print(f"\n  final single-process loss (full dataset) = {baseline_losses[-1]:.6f}")
    print(f"  final FSDP global loss (all 400 samples, gathered weights) = {fsdp_final_global_loss:.6f}")

    close_loss = abs(baseline_losses[-1] - fsdp_final_global_loss) < 1e-2
    real_sharding = local_param_elems < full_param_elems
    print(f"\n{'PASS' if close_loss else 'FAIL'}  FSDP's final global loss matches the single-process baseline's (sharding changes WHERE state lives, not the training math -- same claim zero1/zero2/zero3 verify)")
    print(f"{'PASS' if real_sharding else 'FAIL'}  each rank's LOCAL parameter count is strictly smaller than the full model's -- real sharding measured directly, not assumed from the API name")

    ok = close_loss and real_sharding
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
