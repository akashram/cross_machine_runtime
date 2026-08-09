"""PLAN.md Phase 19 step 6: one real run through a production training
framework. DeepSpeed was attempted FIRST, per PLAN.md's explicit ask, and
hit two real, independent walls (see README.md's Findings) rather than
being skipped on assumption. Ray Train (`ray[train]`, an extras install
on top of the already-installed `ray`, not a new package) is the real,
working fallback: trains the SAME architecture as step 1's
`transformer_torch.py` (token+positional embedding, N pre-LN blocks with
causal MHA + ReLU MLP, final LayerNorm, no-bias output projection),
through Ray Train's real `TorchTrainer`, with `DistributedDataParallel`
wired in automatically by Ray Train (not manually, unlike step 3's
`ddp_gloo.py`).

REAL BUG, caught by running this, not by reading Ray's docs: Ray Train
workers run in SEPARATE actor processes, each with its OWN working
directory -- a driver-side `sys.path.insert(0, "../pytorch_transformer")`
plus `from transformer_torch import ...` has no effect inside those
worker processes (`ModuleNotFoundError: No module named
'transformer_torch'`), even though the identical import works fine when
run standalone in the driver. Rather than fight Ray's `runtime_env`/
`py_modules` module-shipping mechanics further, this file is made
self-contained: the model architecture is duplicated here (NOT imported
from `transformer_torch.py`) so every Ray worker has it locally with no
cross-process import to break. A deliberate, disclosed trade of a little
duplication for robustness, not an oversight.
"""

import math

import ray
import ray.train.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

CORPUS = "the quick fox jumps "
CPP_LOSS_BEFORE = 3.1891
CPP_LOSS_AFTER = 0.0171
NUM_WORKERS = 2
NUM_EPOCHS = 400


class CharTokenizer:
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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        seq, d_model = x.shape
        q = self.wq(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.wk(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.wv(x).view(seq, self.num_heads, self.head_dim).transpose(0, 1)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-1e9"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(0, 1).contiguous().view(seq, d_model)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        x = x + self.w2(F.relu(self.w1(h)))
        return x


class TransformerTorch(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        seq = token_ids.shape[0]
        positions = torch.arange(seq, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.w_out(x)


def next_token_loss(logits, token_ids):
    return F.cross_entropy(logits[:-1], token_ids[1:])


def train_loop_per_worker(config):
    tok = CharTokenizer(CORPUS)
    token_ids = torch.tensor(tok.encode(CORPUS), dtype=torch.long)

    torch.manual_seed(9)  # every worker builds the SAME initial weights
    model = TransformerTorch(vocab_size=tok.vocab_size, d_model=16, num_heads=2, num_layers=2, d_ff=32,
                              max_seq_len=32)
    model = ray.train.torch.prepare_model(model)  # Ray Train wires up real DDP here
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    first_loss = None
    loss = None
    for _ in range(NUM_EPOCHS):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = next_token_loss(logits, token_ids)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()

    ray.train.report({"first_loss": first_loss, "last_loss": loss.item()})


def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=NUM_WORKERS, use_gpu=False),
        run_config=RunConfig(name="phase19_step6_ray_train", storage_path="/tmp/ray_train_phase19"),
    )
    result = trainer.fit()

    first_loss = result.metrics["first_loss"]
    last_loss = result.metrics["last_loss"]
    print(f"\n  Ray Train TorchTrainer, {NUM_WORKERS} real DDP workers (Ray-managed, not manually wired like step 3):")
    print(f"  corpus: {CORPUS!r}")
    print(f"  Ray Train: loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  C++ (transformer_test.cpp, real captured):  loss {CPP_LOSS_BEFORE:.4f} -> {CPP_LOSS_AFTER:.4f}")

    trained_to_low_loss = last_loss < 0.1
    print(f"\n{'PASS' if trained_to_low_loss else 'FAIL'}  Ray Train's final loss < 0.1 (C++ reached 0.0171), a real production-framework training run completed end to end")

    ray.shutdown()
    return 0 if trained_to_low_loss else 1


if __name__ == "__main__":
    raise SystemExit(main())
