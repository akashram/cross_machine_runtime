"""execute_stablehlo.py — load a serialized StableHLO artifact (produced by
step 3's `StableHLOLowerPass`) and execute it via JAX, validating the output
against a plain-numpy reference implementation of the same computation.

PLAN.md Phase 8 step 4: "use JAX to execute the lowered StableHLO, validate
outputs match CPU/GPU reference."

API note: JAX's public entry point for loading a portable StableHLO artifact
is `jax.export` (`jax.export.deserialize(bytes).call(*args)`); this was
`jax.experimental.export` before JAX consolidated it into a stable top-level
module. Pin a JAX version on the TPU VM and confirm which name applies —
this repo's PLAN.md predates that rename settling, so treat the import below
as the intent, not a guaranteed-exact API surface for whatever JAX version
`gcp_setup/provision_tpu_vm.sh` installs.

Unrun here — no JAX/TPU on this Mac, and no real serialized artifact from
step 3 (that needs an MLIR+StableHLO build to produce).
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np


def reference_matmul_bias_relu(a: np.ndarray, b: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Plain-numpy reference for the matmul -> bias-add -> relu chain
    stablehlo_lower's README documents as its first validation row."""
    return np.maximum(np.matmul(a, b) + bias, 0.0)


def reference_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def run_exported_stablehlo(artifact_path: str, *args: jnp.ndarray) -> jnp.ndarray:
    """Deserialize a StableHLO artifact and call it as a JAX function."""
    with open(artifact_path, "rb") as f:
        serialized = f.read()
    exported = jax.export.deserialize(serialized)
    return exported.call(*args)


def validate_matmul_bias_relu(artifact_path: str, n: int = 256, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    bias = rng.standard_normal((n,), dtype=np.float32)

    want = reference_matmul_bias_relu(a, b, bias)
    got = np.asarray(run_exported_stablehlo(artifact_path, jnp.asarray(a), jnp.asarray(b), jnp.asarray(bias)))

    max_abs_err = float(np.max(np.abs(got - want)))
    print(f"matmul_bias_relu {n}x{n}: max_abs_err vs numpy reference = {max_abs_err:.3e}")
    if max_abs_err > 1e-2:
        raise RuntimeError(f"correctness check failed: err={max_abs_err:.3e}")


def validate_softmax(artifact_path: str, n: int = 512, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, n), dtype=np.float32)

    want = reference_softmax(x, axis=-1)
    got = np.asarray(run_exported_stablehlo(artifact_path, jnp.asarray(x)))

    max_abs_err = float(np.max(np.abs(got - want)))
    print(f"softmax 8x{n}: max_abs_err vs numpy reference = {max_abs_err:.3e}")
    if max_abs_err > 1e-2:
        raise RuntimeError(f"correctness check failed: err={max_abs_err:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matmul-bias-relu-artifact", required=True,
                         help="path to serialized StableHLO artifact for the matmul+bias+relu module")
    parser.add_argument("--softmax-artifact", required=True,
                         help="path to serialized StableHLO artifact for the softmax module")
    args = parser.parse_args()

    validate_matmul_bias_relu(args.matmul_bias_relu_artifact)
    validate_softmax(args.softmax_artifact)
    print("execute_stablehlo: PASS")
