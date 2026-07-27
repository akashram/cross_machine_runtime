"""dump_hlo.py — dump XLA HLO (and, where the backend exposes it, the
lowered LLO) for a representative kernel, as the raw material for the VLIW
instruction-bundling analysis in this step's README.

PLAN.md Phase 8 step 10: "inspect XLA-generated HLO for a kernel,
understand the instruction bundling, document how this differs from x86
OOO and NVIDIA SIMT." HLO itself (`.as_text()` below) is backend-agnostic —
it's the same IR XLA would emit compiling for CPU — but the *compiled*
executable's backend-specific structure (what actually maps onto the TPU
vector unit's VLIW bundles) is only inspectable through `jax.jit(...).lower(
...).compile()`'s backend-specific text/proto dump, which requires
compiling for the `tpu` backend specifically; `as_text()` on a CPU-compiled
executable would show LLVM IR structure instead, not VLIW bundles.

Unrun here — no TPU device on this Mac (would print CPU-backend HLO/LLVM
IR instead, not the TPU-specific bundling this step needs).
"""

import jax
import jax.numpy as jnp


def dump_matmul_bias_relu_hlo() -> str:
    def kernel(x, w, b):
        return jnp.maximum(x @ w + b, 0.0)

    x = jnp.zeros((512, 512), dtype=jnp.bfloat16)
    w = jnp.zeros((512, 512), dtype=jnp.bfloat16)
    b = jnp.zeros((512,), dtype=jnp.bfloat16)

    lowered = jax.jit(kernel).lower(x, w, b)
    return lowered.compiler_ir(dialect="hlo").as_hlo_text()


def dump_compiled_backend_text() -> str:
    def kernel(x, w, b):
        return jnp.maximum(x @ w + b, 0.0)

    x = jnp.zeros((512, 512), dtype=jnp.bfloat16)
    w = jnp.zeros((512, 512), dtype=jnp.bfloat16)
    b = jnp.zeros((512,), dtype=jnp.bfloat16)

    compiled = jax.jit(kernel).lower(x, w, b).compile()
    # Backend-specific compiled representation. On the TPU backend this is
    # where VLIW-bundle structure (if exposed at all — Google does not
    # publicly document TPU's VLIW mnemonics) would actually be visible;
    # `as_text()` on other backends shows LLVM IR / PTX / SASS-adjacent
    # text instead.
    return compiled.as_text()


if __name__ == "__main__":
    print(f"backend: {jax.default_backend()}")
    print("=== HLO (backend-agnostic) ===")
    print(dump_matmul_bias_relu_hlo())
    print("\n=== compiled executable text (backend-specific) ===")
    print(dump_compiled_backend_text())
