# Autograd Engine

**Status: STUB — can develop on CPU (macOS or Linux).**

## What this measures
Reverse-mode tape-based autograd for the ops in the runtime dialect.
OR explicit documented decision to interface with PyTorch autograd.
Validate gradient correctness against finite differences.

## Results
TODO: implement and test.

| Op | Max grad error vs finite diff | Test status |
|----|------------------------------|-------------|
| matmul | TODO | TODO |
| relu | TODO | TODO |
| softmax | TODO | TODO |
| cross_entropy | TODO | TODO |

## Hardware notes
- Required: any machine (CPU development first, then validate on GPU)
