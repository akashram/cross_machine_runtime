# Phase 6: Distributed GPU Training

**Status: STUB — requires p4d.24xlarge (8× A100 with NVLink + EFA).**

## Overview
Full distributed training stack: ZeRO optimizer stages, 3D parallelism
(data + tensor + pipeline), MoE, checkpoint sharding, and RLHF pipeline
(SFT → reward model → PPO → DPO).

## Steps

| # | Directory | What | Hardware |
|---|-----------|------|----------|
| 1 | data_loading | WebDataset, multi-worker, sharding | GPU + storage |
| 2 | gpudirect_storage | NVMe → GPU direct | p4d with NVMe |
| 3 | data_parallel | manual gradient all-reduce | multi-GPU |
| 4 | grad_accum | micro-batch accumulation | multi-GPU |
| 5 | grad_clipping | distributed gradient norm + clip | multi-GPU |
| 6 | autograd | reverse-mode tape autograd | any |
| 7 | zero1 | shard optimizer states | multi-GPU |
| 8 | zero2 | + gradient sharding | multi-GPU |
| 9 | zero3 | + parameter sharding | multi-GPU |
| 10 | zero_infinity | CPU/NVMe offload | multi-GPU |
| 11 | col_row_linear | tensor parallel linear layers | multi-GPU |
| 12 | tensor_parallel_attn | tensor parallel attention | multi-GPU |
| 13 | seq_parallel | sequence parallelism | multi-GPU |
| 14 | pipeline_1f1b | 1F1B interleaved pipeline | multi-GPU |
| 15 | parallel_3d | 3D parallelism combination | multi-GPU |
| 16 | moe | MoE expert parallelism | multi-GPU |
| 17 | checkpoint | sharded checkpoint, async write | multi-GPU |
| 18 | compute_comm_overlap | double-buffer backward + all-reduce | multi-GPU |
| 19 | sync_batchnorm | SyncBatchNorm all-reduce | multi-GPU |
| 20 | full_training_loop | end-to-end loop + latency breakdown | multi-GPU |
| 21 | sparsity_training | 2:4 sparsity during training | A100/H100 |
| 22 | sft | supervised fine-tuning | GPU |
| 23 | reward_model | Bradley-Terry reward model | GPU |
| 24 | ppo_rlhf | PPO + KL penalty RLHF | GPU |
| 25 | dpo | Direct Preference Optimization | GPU |

## Hardware notes
- Minimum: p4d.24xlarge (8× A100, 320 GB GPU RAM, 400 Gbps EFA)
- Steps 6, 22-25 can be developed on smaller GPU (g4dn.xlarge) for correctness
