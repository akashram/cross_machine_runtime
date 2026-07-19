# IR Parser Fuzzing (libFuzzer)

**Status: code-complete, not yet built — requires MLIR on Linux with libFuzzer support.**

## What this measures
Fuzzes the runtime dialect IR parser and the full pass pipeline with random-but-valid
IR inputs. Documents any crashes or assertion failures found.

## Design
`LLVMFuzzerTestOneInput` (`ir_fuzz.cpp`) parses the fuzz input as MLIR text
and, on successful parse, runs the entire Phase 4 pipeline (shape inference
→ fusion → affine lowering/tiling → mem planning → remat → placement →
sharding → kernel spec) — the same order `AotCompiler` uses (step 12).
Parse failures on malformed input are expected and not bugs; a registered
diagnostic handler swallows parser error output so it doesn't drown real
crash reports in libFuzzer's log. What this is actually hunting for: a pass
downstream of shape inference indexing into a shape it assumed (e.g. rank
≥ 2) that random IR didn't actually provide.

## Results
TODO: run on Linux with MLIR + libFuzzer.

| Fuzzer | Corpus size | Unique crashes found | Exec/sec |
|--------|-------------|---------------------|----------|
| ir_fuzz (parser) | TODO | TODO | TODO |
| pass_fuzz (pipeline) | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86, Clang with -fsanitize=fuzzer
- Run: `./ir_fuzz -runs=1000000 -max_len=4096 corpus/`
