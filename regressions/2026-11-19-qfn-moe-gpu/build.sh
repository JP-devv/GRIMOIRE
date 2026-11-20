#!/bin/bash
# =====================================================================
#  build.sh  --  compile + link the QFN MoE GPU regression.  HOST ONLY.
#
#  No device image is AOT-compiled here and nothing is executed: the test
#  TU is built for spir64 (JIT) and linked against the objects already in
#  obj-jit/, which is the lane bin/grimoire-jit used.  A full
#  build_b70.sh run would AOT-recompile the whole engine for two dies and
#  take minutes; this takes seconds.
#
#  Run from anywhere:            regressions/2026-11-19-qfn-moe-gpu/build.sh
#  Then, on a free card:         regressions/2026-11-19-qfn-moe-gpu/build/test_qfn_moe_gpu
#  Host-only, no queue at all:   regressions/2026-11-19-qfn-moe-gpu/build/test_qfn_moe_gpu --host-only
# =====================================================================
set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root"

# Intel's vars.sh dereferences OCL_ICD_FILENAMES before assigning it, so
# `set -u` kills the source.  Seed it and keep -u off while sourcing.
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set +u
command -v icpx >/dev/null || { echo "icpx not found; source /opt/intel/oneapi/setvars.sh"; exit 1; }

shopt -s nullglob
jit_objects=(obj-jit/*.o)
if ((${#jit_objects[@]} == 0)); then
    echo 'Build the JIT lane first: obj-jit/*.o is missing.' >&2
    exit 1
fi
if [[ ! -f bin/libgrimoire_gemm.so || ! -f bin/libgrimoire_attn.so ]]; then
    echo 'Build the bridge libraries in bin/ before linking the fixture.' >&2
    exit 1
fi

# Everything in obj-jit except the three translation units that define main().
grouped_obj=${QFN_GEMM_XMX_OBJ:-obj-jit/gemm_xmx.o}
[[ -f $grouped_obj ]] || { printf 'No grouped GEMM object: %s\n' "$grouped_obj" >&2; exit 1; }
OBJS=("$grouped_obj")
for obj in "${jit_objects[@]}"; do
    case ${obj##*/} in grimoire.o|grimoire_main.o|main.o|gemm_xmx.o) continue ;; esac
    OBJS+=("$obj")
done
printf 'objects: %s (grouped GEMM: %s)\n' "${#OBJS[@]}" "$grouped_obj"

mkdir -p "$here/build"
icpx -fsycl \
     -fsycl-targets=spir64 \
     -O2 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=per_kernel \
     -I include -I src \
     "$here/test_qfn_moe_gpu.cpp" "${OBJS[@]}" \
     -Lbin '-Wl,-rpath,$ORIGIN/../../../bin' \
     -Wl,--no-as-needed -lgrimoire_gemm -lgrimoire_attn -Wl,--as-needed \
     -lpthread \
     -o "$here/build/test_qfn_moe_gpu"

echo "built  : $here/build/test_qfn_moe_gpu"
echo
echo "host-only (no GPU, no queue created):"
echo "  $here/build/test_qfn_moe_gpu --host-only"
echo "device run (needs a free B70 and ~1.3 GiB of free device memory):"
echo "  $here/build/test_qfn_moe_gpu"
