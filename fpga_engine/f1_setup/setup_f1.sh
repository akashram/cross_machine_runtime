#!/usr/bin/env bash
# setup_f1.sh — validate an AWS F1 instance is ready for Vitis HLS / Vivado work.
#
# Run this once, right after first SSH-ing into a freshly launched F1 instance
# (f1.2xlarge or f1.4xlarge, FPGA Developer AMI). It does not build or
# synthesize anything — it only proves the toolchain and FPGA card are
# reachable, so every later step in fpga_engine/ can assume a working
# baseline instead of re-deriving it.
#
# TODO: run on F1 hardware. Every check below is untestable off an actual F1
# instance (xbutil, xclmgmt, and the Vitis license all require the real AMI
# and the attached VU9P card), so this script has never executed end-to-end.
set -euo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK:   $*"; }

echo "== 1. Xilinx runtime (XRT) present =="
command -v xbutil >/dev/null 2>&1 || fail "xbutil not on PATH — source /opt/xilinx/xrt/setup.sh"
xbutil --version || fail "xbutil --version failed"
ok "xbutil found"

echo "== 2. FPGA card visible to XRT =="
# 'xbutil examine' enumerates installed cards; on F1 there should be exactly
# one shell-managed device. Grep for a non-empty device list rather than an
# exact string, since shell/xbutil output format changes across XRT versions.
xbutil examine | tee /tmp/xbutil_examine.log
grep -Eq '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' /tmp/xbutil_examine.log \
  || fail "no PCIe device listed by 'xbutil examine' — card not attached or driver not loaded"
ok "FPGA card enumerated"

echo "== 3. AFI (Amazon FPGA Image) / shell status =="
# On F1, the base shell partition must already be loaded before any user
# bitstream can be programmed. fpga-describe-local-image (from the AWS FPGA
# SDK) reports per-slot load status; slot 0 must show state "loaded".
if command -v fpga-describe-local-image >/dev/null 2>&1; then
  fpga-describe-local-image -S 0 | tee /tmp/afi_status.log
  grep -q "loaded" /tmp/afi_status.log || fail "slot 0 AFI not in 'loaded' state"
  ok "slot 0 shell AFI loaded"
else
  fail "fpga-describe-local-image not found — install aws-fpga SDK (github.com/aws/aws-fpga)"
fi

echo "== 4. Vitis HLS toolchain present =="
command -v vitis_hls >/dev/null 2>&1 || fail "vitis_hls not on PATH — source Vitis settings64.sh"
vitis_hls -version || fail "vitis_hls -version failed"
ok "vitis_hls found"

echo "== 5. Vivado present =="
command -v vivado >/dev/null 2>&1 || fail "vivado not on PATH — source Vivado settings64.sh"
vivado -version || fail "vivado -version failed"
ok "vivado found"

echo "== 6. Vitis 'hello world' kernel build + run =="
# Mirrors Xilinx's own vector-add example: build the smallest possible kernel
# for the target platform, run it against XRT, and confirm the returned
# result matches a host-side reference. This is the actual end-to-end proof
# that HLS synthesis -> bitstream -> XRT execution works on this instance,
# not just that individual tools are installed.
WORKDIR="$(mktemp -d)"
PLATFORM="${F1_PLATFORM:-xilinx_aws-vu9p-f1_shell-v04261818_201920_2}"
pushd "$WORKDIR" >/dev/null

git clone --depth 1 https://github.com/Xilinx/Vitis_Accel_Examples.git
pushd Vitis_Accel_Examples/hello_world >/dev/null

make check TARGET=hw PLATFORM="$PLATFORM" DEVICE="$PLATFORM" \
  || fail "hello_world hw build/run failed — see build_dir.hw.*/ logs"

popd >/dev/null
popd >/dev/null
rm -rf "$WORKDIR"
ok "hello_world kernel built and ran correctly against XRT (TARGET=hw)"

echo "== ALL CHECKS PASSED =="
