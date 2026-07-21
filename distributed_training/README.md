# Phase 6: Distributed GPU Training

**Status: 1/25 steps code-complete and locally run (Mac); remaining 24
require multi-GPU hardware (p4d.24xlarge, 8× A100 with NVLink + EFA) and
are still stubs pending local implementation, per CLAUDE.md's execution
order.**

## Overview
Full distributed training stack: ZeRO optimizer stages, 3D parallelism
(data + tensor + pipeline), MoE, checkpoint sharding, and RLHF pipeline
(SFT → reward model → PPO → DPO).

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | data_loading | WebDataset, multi-worker, rank sharding | **built + run** (portable, no GPU needed) |
| 2 | gpudirect_storage | NVMe → GPU direct | stub, needs p4d with NVMe |
| 3 | data_parallel | manual gradient all-reduce | stub, needs multi-GPU |
| 4 | grad_accum | micro-batch accumulation | stub, needs multi-GPU |
| 5 | grad_clipping | distributed gradient norm + clip | stub, needs multi-GPU |
| 6 | autograd | reverse-mode tape autograd | stub, developable on CPU |
| 7 | zero1 | shard optimizer states | stub, needs multi-GPU |
| 8 | zero2 | + gradient sharding | stub, needs multi-GPU |
| 9 | zero3 | + parameter sharding | stub, needs multi-GPU |
| 10 | zero_infinity | CPU/NVMe offload | stub, needs multi-GPU |
| 11 | col_row_linear | tensor parallel linear layers | stub, needs multi-GPU |
| 12 | tensor_parallel_attn | tensor parallel attention | stub, needs multi-GPU |
| 13 | seq_parallel | sequence parallelism | stub, needs multi-GPU |
| 14 | pipeline_1f1b | 1F1B interleaved pipeline | stub, needs multi-GPU |
| 15 | parallel_3d | 3D parallelism combination | stub, needs multi-GPU |
| 16 | moe | MoE expert parallelism | stub, needs multi-GPU |
| 17 | checkpoint | sharded checkpoint, async write | stub, needs multi-GPU |
| 18 | compute_comm_overlap | double-buffer backward + all-reduce | stub, needs multi-GPU |
| 19 | sync_batchnorm | SyncBatchNorm all-reduce | stub, needs multi-GPU |
| 20 | full_training_loop | end-to-end loop + latency breakdown | stub, needs multi-GPU |
| 21 | sparsity_training | 2:4 sparsity during training | stub, needs A100/H100 |
| 22 | sft | supervised fine-tuning | stub, developable on smaller GPU |
| 23 | reward_model | Bradley-Terry reward model | stub, developable on smaller GPU |
| 24 | ppo_rlhf | PPO + KL penalty RLHF | stub, developable on smaller GPU |
| 25 | dpo | Direct Preference Optimization | stub, developable on smaller GPU |

## Hardware notes
- Minimum: p4d.24xlarge (8× A100, 320 GB GPU RAM, 400 Gbps EFA)
- Steps 6, 22-25 can be developed on smaller GPU (g4dn.xlarge) for correctness
