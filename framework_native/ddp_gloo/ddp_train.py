"""PLAN.md Phase 19 step 3: real torch.nn.parallel.DistributedDataParallel
training over CPU (`gloo` backend), across REAL separate OS processes
(`torch.multiprocessing.spawn` -- genuine multi-process, not threads),
diffed directly against `distributed_training/data_parallel`'s own
hand-written all-reduce implementation and its exact task setup: a
synthetic linear regression `y = X.w_true + noise` with a KNOWN true
weight vector, full-batch (not mini-batch) gradient descent, so the
comparison is against a known-correct closed-form reference the same way
`distributed_training/data_parallel/README.md`'s own sanity run is.

Each rank owns an equal contiguous shard of the dataset (matching the C++
version's sharding), computes its local MEAN loss over its shard, and
DDP's default gradient all-reduce AVERAGES gradients across ranks --
mathematically identical to a single-process full-batch mean-loss
gradient over the union of all shards, for equal-sized shards. Compared
against a real single-process baseline trained the same way, same
initial weights, same number of steps.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


NUM_SAMPLES = 400
NUM_FEATURES = 8
NOISE_STD = 0.1
NUM_STEPS = 60
LR = 0.05
WORLD_SIZE = 4


def make_dataset(seed: int, w_true: torch.Tensor):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NUM_SAMPLES, NUM_FEATURES, generator=g)
    noise = torch.randn(NUM_SAMPLES, generator=g) * NOISE_STD
    y = X @ w_true + noise
    return X, y


def mse(model, X, y):
    pred = model(X).squeeze(-1)
    return ((pred - y) ** 2).mean()


def train_single_process(w_true, init_state):
    X, y = make_dataset(seed=7, w_true=w_true)
    model = nn.Linear(NUM_FEATURES, 1, bias=False)
    model.load_state_dict(init_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    losses = []
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        loss = mse(model, X, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, model.state_dict()["weight"].detach().clone()


def ddp_worker(rank, world_size, w_true, init_state, result_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    X_full, y_full = make_dataset(seed=7, w_true=w_true)
    shard = NUM_SAMPLES // world_size
    start = rank * shard
    end = start + shard
    X, y = X_full[start:end], y_full[start:end]

    model = nn.Linear(NUM_FEATURES, 1, bias=False)
    model.load_state_dict(init_state)
    ddp_model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=LR)

    losses = []
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        loss = mse(ddp_model, X, y)
        loss.backward()  # DDP all-reduces (averages) gradients across ranks here
        optimizer.step()
        losses.append(loss.item())

    if rank == 0:
        # rank 0's shard-local loss isn't the global loss; report the
        # GLOBAL full-batch loss under the final synchronized weights
        # (identical on every rank after the last all-reduce) for a
        # true apples-to-apples comparison against the single-process run.
        with torch.no_grad():
            final_global_loss = mse(model, X_full, y_full).item()
        result_queue.put((losses, model.state_dict()["weight"].detach().clone(), final_global_loss))

    dist.destroy_process_group()


def main():
    torch.manual_seed(3)
    w_true = torch.randn(NUM_FEATURES)
    init_model = nn.Linear(NUM_FEATURES, 1, bias=False)
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}

    baseline_losses, baseline_w = train_single_process(w_true, init_state)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=ddp_worker, args=(rank, WORLD_SIZE, w_true, init_state, result_queue))
        p.start()
        processes.append(p)
    ddp_losses, ddp_w, ddp_final_global_loss = result_queue.get()
    for p in processes:
        p.join()

    print(f"  {WORLD_SIZE} REAL separate OS processes (torch.multiprocessing spawn, gloo backend)")
    print(f"  step   baseline_loss   ddp_rank0_shard_loss")
    for i in (0, 10, 20, 30, 40, 50, NUM_STEPS - 1):
        print(f"  {i:4d}   {baseline_losses[i]:.6f}      {ddp_losses[i]:.6f}")

    w_diff = (baseline_w - ddp_w).abs().max().item()
    print(f"\n  final single-process loss (full dataset)     = {baseline_losses[-1]:.6f}")
    print(f"  final DDP global loss (all 400 samples, synced weights) = {ddp_final_global_loss:.6f}")
    print(f"  max |weight difference| baseline vs. DDP = {w_diff:.6f}")

    close_loss = abs(baseline_losses[-1] - ddp_final_global_loss) < 1e-3
    close_weights = w_diff < 1e-3
    print(f"\n{'PASS' if close_loss else 'FAIL'}  DDP's final global loss matches the single-process baseline's to within 1e-3")
    print(f"{'PASS' if close_weights else 'FAIL'}  DDP's final weights match the single-process baseline's to within 1e-3 (max abs diff)")

    ok = close_loss and close_weights
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
