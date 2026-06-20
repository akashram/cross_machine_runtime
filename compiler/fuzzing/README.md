# IR Parser Fuzzing (libFuzzer)

**Status: STUB — requires MLIR on Linux with libFuzzer support.**

## What this measures
Fuzzes the runtime dialect IR parser and the full pass pipeline with random-but-valid
IR inputs. Documents any crashes or assertion failures found.

## Results
TODO: run on Linux with MLIR + libFuzzer.

| Fuzzer | Corpus size | Unique crashes found | Exec/sec |
|--------|-------------|---------------------|----------|
| ir_fuzz (parser) | TODO | TODO | TODO |
| pass_fuzz (pipeline) | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86, Clang with -fsanitize=fuzzer
- Run: `./ir_fuzz -runs=1000000 -max_len=4096 corpus/`
