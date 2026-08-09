"""PLAN.md Phase 19 step 5: a `jit`/`grad`/`vmap`/`pmap` JAX port of the
same transformer architecture step 1 ported to PyTorch (and
transformer/transformer_model.h implements in hand-derived C++).

Scope note: pure `jax`/`jax.numpy` (a params pytree = a plain nested
dict of arrays, manual SGD update via `jax.tree_util.tree_map`), no
Flax/Optax -- neither is installed (`.venv` only has the `jax`/`jaxlib`
already approved and installed for Phase 8's `tpu_engine`), and pure JAX
is enough to demonstrate the `jit`/`grad`/`vmap`/`pmap` functional-
transform model this step is actually about. A disclosed scope choice,
not an oversight.

Architecturally identical to transformer_torch.py / transformer_model.h:
token + positional embedding, N pre-LN blocks (causal MHA + ReLU MLP),
final LayerNorm, no-bias output projection.
"""

import jax
import jax.numpy as jnp
import numpy as np


class CharTokenizer:
    """Same first-occurrence-order algorithm as char_tokenizer.h /
    transformer_torch.py's CharTokenizer."""

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
        return "".join(self.id_to_char[int(i)] for i in ids)


def init_params(key, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
    keys = jax.random.split(key, 4 + num_layers * 8)
    ki = iter(keys)
    stddev = jnp.sqrt(2.0 / d_model)

    def dense(k, fan_in, fan_out):
        return jax.random.normal(k, (fan_in, fan_out)) * stddev

    params = {
        "token_emb": jax.random.normal(next(ki), (vocab_size, d_model)) * 0.02,
        "pos_emb": jax.random.normal(next(ki), (max_seq_len, d_model)) * 0.02,
        "blocks": [],
        "final_gamma": jnp.ones((d_model,)),
        "final_beta": jnp.zeros((d_model,)),
        "w_out": dense(next(ki), d_model, vocab_size),
    }
    for _ in range(num_layers):
        block = {
            "gamma1": jnp.ones((d_model,)), "beta1": jnp.zeros((d_model,)),
            "wq": dense(next(ki), d_model, d_model), "wk": dense(next(ki), d_model, d_model),
            "wv": dense(next(ki), d_model, d_model), "wo": dense(next(ki), d_model, d_model),
            "gamma2": jnp.ones((d_model,)), "beta2": jnp.zeros((d_model,)),
            "w1": dense(next(ki), d_model, d_ff), "b1": jnp.zeros((d_ff,)),
            "w2": dense(next(ki), d_ff, d_model), "b2": jnp.zeros((d_model,)),
        }
        params["blocks"].append(block)
    return params


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def causal_attention(x, block, num_heads):
    seq, d_model = x.shape
    head_dim = d_model // num_heads
    q = (x @ block["wq"]).reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    k = (x @ block["wk"]).reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    v = (x @ block["wv"]).reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    scale = 1.0 / jnp.sqrt(head_dim)
    scores = jnp.einsum("hid,hjd->hij", q, k) * scale
    causal_mask = jnp.triu(jnp.ones((seq, seq), dtype=bool), k=1)
    scores = jnp.where(causal_mask, -1e9, scores)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hij,hjd->hid", attn, v)
    out = out.transpose(1, 0, 2).reshape(seq, d_model)
    return out @ block["wo"]


def block_forward(x, block, num_heads):
    x = x + causal_attention(layer_norm(x, block["gamma1"], block["beta1"]), block, num_heads)
    h = layer_norm(x, block["gamma2"], block["beta2"])
    h = jax.nn.relu(h @ block["w1"] + block["b1"]) @ block["w2"] + block["b2"]
    return x + h


def model_forward(params, token_ids, num_heads):
    seq = token_ids.shape[0]
    x = params["token_emb"][token_ids] + params["pos_emb"][:seq]
    for block in params["blocks"]:
        x = block_forward(x, block, num_heads)
    x = layer_norm(x, params["final_gamma"], params["final_beta"])
    return x @ params["w_out"]


def next_token_loss(params, token_ids, num_heads):
    logits = model_forward(params, token_ids, num_heads)
    log_probs = jax.nn.log_softmax(logits[:-1], axis=-1)
    targets = token_ids[1:]
    return -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])


def greedy_generate(params, tok, prompt_ids, num_new_tokens, num_heads):
    ids = list(prompt_ids)
    for _ in range(num_new_tokens):
        logits = model_forward(params, jnp.array(ids), num_heads)
        next_id = int(jnp.argmax(logits[-1]))
        ids.append(next_id)
    return tok.decode(np.array(ids))
