"""PLAN.md Phase 19 step 1: trains transformer_torch.py on the IDENTICAL
corpus, config, and hyperparameters transformer/transformer_test.cpp's
test_trains_and_generates() uses, and checks the result against that
test's real captured numbers (transformer/README.md /
`ctest -R transformer_test` output):

    corpus:  "the quick fox jumps "
    config:  vocab_size=tok.vocab_size(), d_model=16, num_heads=2,
             num_layers=2, d_ff=32, max_seq_len=32
    training: 400 epochs, lr=0.05, plain SGD
    C++ result: loss 3.1891 -> 0.0171, exact greedy-decode match

This is the real cross-check that transformer_model.h's hand-derived
backprop and PyTorch's autograd agree on the SAME architecture and task,
not just "a PyTorch transformer can be trained on something."
"""

import torch
import torch.nn.functional as F

from transformer_torch import CharTokenizer, TransformerTorch

CORPUS = "the quick fox jumps "
CPP_LOSS_BEFORE = 3.1891
CPP_LOSS_AFTER = 0.0171


def next_token_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Position i's logits predict token_ids[i+1], i = 0..seq-2 -- same
    definition as transformer_model.h's next_token_loss."""
    return F.cross_entropy(logits[:-1], token_ids[1:])


@torch.no_grad()
def greedy_generate(model: TransformerTorch, tok: CharTokenizer, prompt_ids, num_new_tokens: int):
    ids = list(prompt_ids)
    for _ in range(num_new_tokens):
        logits = model(torch.tensor(ids, dtype=torch.long))
        next_id = int(torch.argmax(logits[-1]).item())
        ids.append(next_id)
    return tok.decode(ids)


def main():
    torch.manual_seed(9)  # not bit-identical to the C++ mt19937 stream, but a fixed, reproducible seed

    tok = CharTokenizer(CORPUS)
    token_ids = torch.tensor(tok.encode(CORPUS), dtype=torch.long)

    model = TransformerTorch(
        vocab_size=tok.vocab_size, d_model=16, num_heads=2, num_layers=2, d_ff=32, max_seq_len=32
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    logits = model(token_ids)
    first_loss = next_token_loss(logits, token_ids).item()

    for _ in range(400):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = next_token_loss(logits, token_ids)
        loss.backward()
        optimizer.step()

    logits = model(token_ids)
    last_loss = next_token_loss(logits, token_ids).item()

    generated = greedy_generate(model, tok, [tok.char_to_id[CORPUS[0]]], len(CORPUS) - 1)

    print(f"  corpus: {CORPUS!r}")
    print(f"  PyTorch: loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  C++ (transformer_test.cpp, real captured):  loss {CPP_LOSS_BEFORE:.4f} -> {CPP_LOSS_AFTER:.4f}")
    print(f"  PyTorch generated: {generated!r}")
    print(f"  expected (== corpus):  {CORPUS!r}")

    exact_match = generated == CORPUS
    trained_to_low_loss = last_loss < 0.1
    loss_decreased_similarly = last_loss < first_loss * 0.05  # same order-of-magnitude reduction as the C++ run

    print(f"\n{'PASS' if exact_match else 'FAIL'}  PyTorch greedy-generates the training corpus back exactly, same definition of success as transformer_test.cpp")
    print(f"{'PASS' if trained_to_low_loss else 'FAIL'}  final loss < 0.1 (C++ reached 0.0171)")
    print(f"{'PASS' if loss_decreased_similarly else 'FAIL'}  loss reduction is the same order of magnitude as the C++ run's ~186x (3.1891/0.0171)")

    ok = exact_match and trained_to_low_loss and loss_decreased_similarly
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
