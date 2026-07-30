# Phase 16 step 1 — Dockerfile for the portable build subset.
#
# Status: real, complete, UNRUN (no Docker installed on the dev Mac this
# was written on — see containers/README.md). Written to the same standard
# as this repo's other unrun-but-real artifacts (e.g. fpga_engine/vitis_ai's
# mlp_model.py): every package and command below was chosen by actually
# tracing this tree's CMakeLists.txt files, not guessed.
#
# What "portable" means here, traced not assumed:
#   - Root CMakeLists.txt gates gpu_engine/ behind check_language(CUDA) and
#     compiler/'s MLIR-dependent passes behind `if(DEFINED MLIR_DIR)` —
#     both already skip cleanly with neither toolchain present.
#   - compiler/cost_model/ has zero MLIR dependency and is added
#     unconditionally (see its own CMakeLists.txt comment).
#   - tpu_engine/'s only CMake target (stablehlo_lower) self-gates on
#     `MLIR_DIR AND STABLEHLO_DIR`; everything else in tpu_engine/ is
#     Python/JAX run directly, no CMake target at all.
#   - fpga_engine/'s CMakeLists.txt only warns (not fails) when
#     XILINX_VITIS is unset; every step's CMake target is a portable C++
#     model/testbench, not the HLS/TCL flow itself.
#   - Every OTHER optional dependency in the tree (Protobuf, gRPC,
#     FlatBuffers, libfabric, libbpf) is found via `find_package(... QUIET)`
#     / `find_library` + a graceful `return()` if missing — grep confirms
#     the ONLY unguarded `REQUIRED` finds in the whole tree are
#     find_package(Threads REQUIRED) (POSIX threads, present on any Linux
#     base image) plus CUDAToolkit/MLIR/Stablehlo, all three already gated
#     one level up as above. vcpkg.json has zero declared dependencies.
#   - Net effect: this tree configures and builds successfully on a bare
#     Linux box with nothing but a C++23 compiler + CMake >= 3.25 + Ninja.
#     Installing the optional dev packages below (protobuf/gRPC/
#     FlatBuffers/libfabric/libbpf) doesn't change that baseline — it just
#     turns ON five additional Linux-only targets that otherwise silently
#     skip (see containers/README.md's table): networking/grpc_control,
#     networking/flatbuffers_data, networking/af_xdp,
#     networking/userspace_net, networking/rdma_onesided.
#
# A likely real root cause for .github/workflows/ci.yml's long-standing
# failures (documented there as "consistent with dying at dependency-
# install or configure time"): CMakeLists.txt pins
# `cmake_minimum_required(VERSION 3.25)` (CMakePresets.json matches), but
# ci.yml's `apt-get install ninja-build clang` never installs or pins
# cmake itself — it relies entirely on whatever cmake ships preinstalled
# on GitHub's `ubuntu-latest` runner image at a given point in time, which
# has, across ubuntu-latest's actual Ubuntu-version transitions,
# sometimes been below 3.25 (Ubuntu 22.04's own apt cmake package is
# 3.22.x). That would fail every job in exactly the "fails fast, at
# configure time" way described, with zero relation to any of the
# gRPC/FlatBuffers/libfabric/libbpf/MLIR toolchains ci.yml's own comment
# blames — this fork could not confirm this against the actual GitHub
# Actions run logs (no `gh` CLI available in this environment), so it is
# presented as a well-evidenced hypothesis, not a confirmed diagnosis. This
# Dockerfile sidesteps the question entirely by installing a modern cmake
# explicitly via pip rather than trusting any base image's apt cmake
# version.

# ---- Stage 1: builder ----
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Base toolchain: clang (matches CLAUDE.md's "GCC/clang on Linux cloud
# instances" tooling decision), ninja, python3+pip for a pinned modern
# cmake (see rationale above — do not rely on apt's cmake package).
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    ninja-build \
    python3 \
    python3-pip \
    pkg-config \
    git \
    ca-certificates \
    && pip3 install --break-system-packages --no-cache-dir "cmake>=3.28" \
    && rm -rf /var/lib/apt/lists/*

# Optional dev packages that turn on the five additional Linux-only
# targets named above. None of these are REQUIRED anywhere in the tree —
# every CMakeLists.txt that touches them degrades gracefully — but
# installing them for real is the honest way to demonstrate the "portable
# build" actually covers the Linux-specific pieces this Mac can never
# build, not just the subset that already worked locally too.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libflatbuffers-dev \
    flatbuffers-compiler \
    libfabric-dev \
    libbpf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Same preset/build/test invocation as .github/workflows/ci.yml's Linux
# job (debug preset) and CLAUDE.md's documented workflow — deliberately
# not a new, container-specific build recipe.
RUN cmake --preset debug -DCMAKE_CXX_COMPILER=clang++
RUN cmake --build --preset debug
# Deliberately NOT run at image build time: ctest here would bake this
# Mac-authored image's test PASS/FAIL into a cached Docker layer, and
# some of this repo's own tests (e.g. TSan-instrumented ones under other
# presets) are meaningfully slower / not this preset's job to cover.
# `docker run` (see containers/README.md) runs ctest at container
# start instead, so every run reflects that run, not build time.

# ---- Stage 2: runtime ----
# Deliberately still Ubuntu 24.04 (not a distroless/slim base): this
# image's job is reproducing `ctest`'s full suite, which needs the same
# libstdc++/runtime pieces as the builder, not a minimal single-binary
# deploy target — see containers/README.md for why this differs from,
# say, a hypothetical slim `serving_backend`-only image.
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libprotobuf32t64 \
    libgrpc++1.51 \
    libflatbuffers2 \
    libfabric1 \
    libbpf1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src /src
WORKDIR /src

# `ctest --preset debug` re-discovers and re-runs every registered test
# from the build tree copied in above — this is what `docker run
# <image>` executes by default.
CMD ["ctest", "--preset", "debug", "--output-on-failure"]
