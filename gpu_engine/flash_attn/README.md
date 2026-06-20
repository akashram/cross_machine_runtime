# Flash Attention (Forward + Backward)

**Status: STUB — requires CUDA GPU (A100 recommended). Run on p3.2xlarge or p4d.24xlarge.**

## What this measures
Flash Attention forward kernel: tiled SRAM implementation of scaled dot-product
attention with online softmax and optional causal masking (Dao et al. 2022).
Flash Attention backward: recomputation of attention weights from output to avoid
storing the full attention matrix. Benchmark vs cuDNN attention and naive attention.

## Implementation notes

### Forward kernel
- Tile Q, K, V into SRAM blocks — never materialize full S=QKᵀ or P=softmax(S) in HBM
- Online softmax: maintain running max and normalizer per row (Milakov & Gimelshein 2018)
- Causal masking: set S[i,j] = -inf for j > i before softmax
- Block size (Br, Bc): tune to fill SRAM (A100: 96KB SRAM per SM)
- Memory: O(seq_len) HBM instead of O(seq_len²) for naive

### Backward kernel
- Recompute S = QKᵀ and P = softmax(S) from Q, K, O, l, m (saved statistics)
- Avoids storing the attention matrix — halves backward memory vs naive
- Numerical validation: compare dQ, dK, dV against PyTorch autograd reference

## Results

TODO: run on GPU hardware and fill in this table.

### Forward pass (seq_len=2048, d_head=64, batch=16, FP16)

| Implementation | Latency (ms) | Memory (GB) | FLOPS util (%) |
|----------------|-------------|-------------|-----------------|
| Naive attention | TODO | TODO | TODO |
| cuDNN attention | TODO | TODO | TODO |
| Flash Attention (ours) | TODO | TODO | TODO |

### Backward pass

| Implementation | Latency (ms) | Memory (GB) |
|----------------|-------------|-------------|
| Naive backward | TODO | TODO |
| Flash Attention backward | TODO | TODO |

## Hardware notes
- Required: A100 (for large SRAM) or V100. FP16 for Tensor Core path.
- Build preset: cuda (Linux)
- Numerical validation required before reporting speedup
