# data_reduction

**Status: code-complete AND locally run — pure CPU, real trained model,
real zlib compression, no external dependency.**

## What this measures

PLAN.md Phase 21 step 7: ACTUAL compression ratios on this repo's own
real AI-workload artifacts — a real trained `transformer/` model's
checkpoint weight (`w_out`), its real tokenized training data, its real
raw-text training corpus, and its real intermediate embeddings
(`final_ln_out`) — not an assumed uniform "global data reduction" ratio.
Uses real zlib (`compress2`, `Z_BEST_COMPRESSION`) — a system library
already present via Xcode CLT on macOS and effectively universal on
Linux, not a new local install.

## Results (captured 2026-09-16, Apple clang 14, `--preset release`, this Mac)

```
== Data reduction on real repo artifacts (zlib -9) ==

  trained checkpoint weight (w_out) raw=     3456 bytes  compressed=     3245 bytes  ratio=1.065x
  raw text training corpus     raw=      176 bytes  compressed=       54 bytes  ratio=3.259x
  tokenized training data (int32) raw=      704 bytes  compressed=       86 bytes  ratio=8.186x
  intermediate embeddings (final_ln_out) raw=    22528 bytes  compressed=    20909 bytes  ratio=1.077x
PASS  raw text compresses better than trained float weights -- confirms 'global data reduction' is NOT one uniform ratio across AI-workload artifact types, the real finding PLAN.md step 7 asks for instead of an assumed uniform number
PASS  trained checkpoint weights compress poorly (close to their raw size) -- consistent with post-training float weights carrying close to full bit-level entropy, not the highly-redundant structure a generic compressor exploits

PASS
```

## Findings

- **A real, measured, non-uniform compression profile across four
  distinct AI-workload artifact types**, ranging from 1.065x (trained
  weights) to 8.186x (tokenized training data) — over 7.5x variation in
  reduction ratio depending on WHICH artifact a storage system's "global
  data reduction" claim is actually measured on. A vendor's single
  headline reduction ratio is therefore, at best, a blend across a mix of
  artifact types the customer's specific workload may not match.
- **Trained float weights compress the WORST of the four (1.065x)** —
  consistent with the well-known property that post-training neural
  network weights carry close to full bit-level entropy (no highly
  redundant structure for a generic byte-level compressor like zlib to
  exploit), the opposite of the highly-compressible natural-language text
  vendor demos are often shown on.
- **Tokenized training data compresses BEST (8.186x)** — a small,
  low-cardinality token alphabet (this toy corpus's character-level
  vocabulary) repeated many times is exactly the kind of redundant,
  low-entropy structure zlib's LZ77+Huffman scheme is built for; real
  large-vocabulary subword tokenization on a production-scale corpus
  would likely land somewhere between this toy result and the raw-text
  result, not necessarily reproduce either extreme.
- **Intermediate embeddings compress almost as poorly as trained weights
  (1.077x)** — continuous post-forward-pass activations, like trained
  weights, don't carry the kind of byte-level redundancy generic
  compression exploits, a third genuinely different artifact-type
  behavior from the two above.

## Hardware notes

None — pure CPU, real zlib. The trained model is a real, small
(toy-scale) transformer, not a production-scale checkpoint — the
QUALITATIVE finding (compression ratio varies sharply by artifact type,
and trained weights/activations compress far worse than text) is
consistent with the well-known general property of trained neural
network weights, but the SPECIFIC ratios above are this toy model's,
not a production-scale one's. See `hpc_storage/DESIGN.md`.
