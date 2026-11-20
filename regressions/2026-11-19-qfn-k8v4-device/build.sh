#!/bin/bash
# =====================================================================
#  build.sh  --  compile + link the K8/V4 dense-decode DEVICE PROTOTYPE.
#
#  PROTOTYPE, not a runtime component: one self-contained translation unit,
#  no engine objects, no bridge libraries, nothing from the runtime linked.
#
#  This script only BUILDS.  Device code is emitted for spir64 and JIT-compiled
#  at run time, so the build itself needs no GPU and touches no render node.
#  Running the binary needs one free card; the parent schedules that.
#
#  Build:                 regressions/2026-11-19-qfn-k8v4-device/build.sh
#  Reference only:        .../build/test_qfn_k8v4_device --host-only
#  Enumerate cards:       .../build/test_qfn_k8v4_device --list-devices
#  Device run:            .../build/test_qfn_k8v4_device --device <name-substr>
# =====================================================================
set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Intel's vars.sh dereferences OCL_ICD_FILENAMES before assigning it, so
# `set -u` would kill the source.  Seed it and keep -u off while sourcing.
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set +u
command -v icpx >/dev/null || { echo "icpx not found; source /opt/intel/oneapi/setvars.sh" >&2; exit 1; }

# The binary should run without a sourced oneAPI environment, so bake an rpath
# to the SYCL runtime next to the icpx that built it (no hard-coded version).
icpx_bin="$(readlink -f "$(command -v icpx)")"
sicl_lib="$(cd "$(dirname "$icpx_bin")/../lib" && pwd)"

mkdir -p "$here/build"
icpx -fsycl \
     -fsycl-targets=spir64 \
     -O2 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=per_kernel \
     -Wall -Wextra -Werror \
     -Wno-deprecated-declarations \
     "$here/test_qfn_k8v4_device.cpp" \
     -Wl,-rpath,"$sicl_lib" \
     -o "$here/build/test_qfn_k8v4_device"

echo "sycl rt : $sicl_lib (rpath baked in)"

echo "built  : $here/build/test_qfn_k8v4_device"
echo
echo "reference only, no queue, safe on any host:"
echo "  $here/build/test_qfn_k8v4_device --host-only"
echo "enumerate cards (no queue, no compute):"
echo "  $here/build/test_qfn_k8v4_device --list-devices"
echo "device run (needs one free B70, no more than ~1 MiB of device memory):"
echo "  $here/build/test_qfn_k8v4_device --device B70"
