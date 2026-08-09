"""PLAN.md Phase 19 step 1: a real torch.nn.Module reimplementation of
transformer/transformer_model.h, architecturally identical: token +
learned positional embedding, N pre-LN decoder blocks (causal multi-head
self-attention + residual, ReLU MLP + residual), final LayerNorm, linear
output projection (no bias) to vocab logits. Same char-level tokenizer
algorithm as transformer/char_tokenizer.h (build vocab in first-occurrence
order scanning the corpus left to right -- not sorted).

This file is the cross-check that the hand-derived C++ math in
transformer/transformer_model.h and PyTorch's real autograd agree, not
just "PyTorch training works" -- see train_and_compare.py for the actual
comparison against transformer_test.cpp's real captured numbers.

No batch dimension (batch=1, one sequence per forward call) -- same
scope choice transformer_model.h itself makes, kept here for an
apples-to-apples architectural match, not a PyTorch limitation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """Matches transformer/char_tokenizer.h exactly: vocab built in
    first-occurrence order over the corpus, not sorted."""

    def __init__(self, corpus: str):
        self.char_to_id = {}
        self.id_to_char = []
        for c in corpus:
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.id_to_char)
                self.id_to_char.append(c)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str):
        return [self.char_to_id[c] for c in text]

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, matching
    transformer_model.h's causal_attention_forward per head (score mask
    set to -inf above the diagonal before softmax) composed across heads
    via wq/wk/wv/wo, same as BlockParams."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, d_model = x.shape
        q = self.wq(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, seq, head_dim]
        k = self.wk(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.wv(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [heads, seq, seq]
        causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-1e9"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [heads, seq, head_dim]
        out = out.transpose(0, 1).contiguous().view(seq, d_model)
        return self.wo(out)


class MLP(nn.Module):
    """W2 @ ReLU(W1 @ x + b1) + b2, matching block_forward's MLP exactly
    (ReLU, not GELU -- transformer_model.h's actual activation)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class Block(nn.Module):
    """Pre-LN block: x = x + attn(LN1(x)); x = x + mlp(LN2(x))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerTorch(nn.Module):
    """Token + positional embedding, N pre-LN blocks, final LayerNorm,
    linear (no-bias) output projection to vocab logits -- architecturally
    identical to transformer/transformer_model.h's ModelParams/
    model_forward."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq = token_ids.shape[0]
        positions = torch.arange(seq, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.w_out(x)
