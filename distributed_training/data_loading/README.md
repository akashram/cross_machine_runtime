# Distributed Data Loading

**Status: code-complete AND locally run — portable, pure C++, no CUDA/Linux
dependency.**

## What this measures

PLAN.md Phase 6 step 1: distributed data loading. Three pieces:
1. A minimal WebDataset codec (`webdataset_shard.h/.cpp`) — USTAR tar
   reader/writer, groups adjacent tar entries sharing a key into samples.
2. Rank sharding (`data_loader.h/.cpp`) — dataset shard files striped
   across `world_size` ranks by index (`i % world_size`), so ranks never
   coordinate mid-epoch.
3. Multi-worker prefetch (`data_loader.cpp` + `prefetch_queue.h`) — N
   worker threads per rank pull shards from a shared atomic cursor and
   push decoded samples onto a bounded blocking queue for the training
   loop to consume.

PLAN.md's target metric is "GPU utilization with and without pipeline."
There's no GPU on this Mac, so the substitute measured here is **consumer
stall fraction**: the fraction of `next()` calls where the prefetch queue
was empty (i.e., where a real training loop would have stalled waiting for
data instead of feeding the GPU). That's the same phenomenon GPU
utilization would show — a GPU sitting idle waiting for data starves for
exactly the same reason the consumer thread here does.

## Design

- **Shard-granularity sharding, not sample-granularity**: rank r owns
  `shard_paths[i]` for every `i % world_size == r`. This needs zero
  cross-rank coordination — every rank computes its own assignment from
  `shard_paths.size()` alone — at the cost of requiring `num_shards >>
  world_size` for the striping to stay balanced (the standard WebDataset
  guidance; see `data_loader.h`).
- **Dynamic work-stealing across shards within a rank**: workers claim
  shards via `std::atomic<size_t> next_shard_idx_.fetch_add(1)` rather
  than a static split, because shards vary in sample count and a static
  split would idle some workers while others are still decoding.
- **Bounded blocking queue, not lock-free**: `PrefetchQueue` is
  mutex+condvar, deliberately not one of foundation/'s lock-free ring
  buffers. A decode-and-push cycle costs microseconds to milliseconds, so
  contention on one mutex is noise — PyTorch's DataLoader makes the same
  call for the same reason. See `prefetch_queue.h`.
- **Missing/corrupt shard tolerance**: a shard that fails to open is
  skipped, not fatal (`data_loader.cpp`'s `worker_loop`) — one bad shard
  file shouldn't kill a multi-hour training run.

## Sanity-run output (Mac, local disk, 2026-07-21)

`data_loader_test`: synthetic 62MB WebDataset (20 shards x 100 samples x
32KB payload). Correctness: 4-way rank sharding is disjoint and complete,
every sample's payload round-trips byte-for-byte through the tar codec.
Bench: drain the full dataset with `num_workers=1` vs `num_workers=8`,
tracking wall-clock and how often the consumer found the queue empty.

```
=== correctness: 4-way rank sharding, disjoint + complete ===
PASS

=== bench: single-worker vs 8-worker prefetch (62 MB dataset) ===
num_workers=1: 0.804s  2488 samples/s  81.5 MB/s  stall_fraction=0.789
num_workers=8: 0.746s  2682 samples/s  87.9 MB/s  stall_fraction=0.001
speedup: 1.08x

PASS
```

**Interpretation**: raw wall-clock throughput barely moves (1.08x) —
expected, since both configurations read the same total bytes off one
local SSD and disk bandwidth, not CPU decode, is the bottleneck here.
What does move is stall fraction: 78.9% of `next()` calls found an empty
queue with one worker, vs 0.1% with eight. That's the actual point of
prefetching — it's a latency-hiding technique, not a throughput one, when
the pipeline is I/O-bound with slack CPU. On a real training run where
the consumer is a GPU step (milliseconds, done in parallel with the CPU
decode threads) rather than a tight re-loop, one slow worker means the
GPU sits idle for exactly the stall fraction shown above; eight workers
keep the queue full enough that it doesn't.

## Results
TODO: run on GPU hardware — the number that matters is measured GPU
utilization (`nvidia-smi dmon` or Nsight Systems trace) during a real
training step loop, not the CPU-only stall-fraction proxy above.

| Setup | GPU util, no prefetch | GPU util, 8-worker prefetch |
|-------|----------------------|------------------------------|
| Single node, real dataset | TODO | TODO |

## Hardware notes
- This step: none required — runs anywhere with a filesystem and threads.
- GPU-utilization validation (Results table above): GPU instance, real
  training step loop consuming from this loader.
