#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
work=$PWD
base=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP
stage=inputs
exec > split-build.log 2>&1
trap 'rc=$?; sha256sum -c "$base/serving.sha256" || rc=1; printf "stage=%s exit=%d\n" "$stage" "$rc" > split-build.receipt; printf "%d\n" "$rc" > split-build.exit; exit "$rc"' EXIT
[[ $(< build.receipt) == 'stage=verify exit=0' ]]
sha256sum -c candidate.sha256
sha256sum -c base-objects.sha256
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force
objects=()
for obj in "$base"/obj/*.o; do
    case ${obj##*/} in grimoire.cpp.o|attention.cpp.o) ;; *) objects+=("$obj");; esac
done
stage=host-regression
g++ -std=c++20 -Wall -Wextra -Werror -Isrc test_split_plan.cpp -o test_split_plan
./test_split_plan
printf 'QFN verifier split plan PASS\n'
flags=(-fsycl -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -O2 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -Iinclude -Isrc -Ixe2)
stage=verifier
icpx "${flags[@]}" -c verifier-split.cpp -o split-grimoire.cpp.o
stage=link
icpx "${flags[@]}" -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -fsycl-max-parallel-link-jobs=4 split-grimoire.cpp.o attention.cpp.o "${objects[@]}" -pthread -L. \
    -lgrimoire_gemm -lgrimoire_attn '-Wl,-rpath,$ORIGIN' -o qfn-verify-split
stage=verify
strings qfn-verify-split | awk '/GRIMOIRE_QFN_VERIFY_SPLIT_KV/{selector=1} /QFN verifier KV:/{trace=1} END {exit !(selector&&trace)}'
sha256sum "$work"/{qfn-verify-split,verifier-split.cpp,src/qfn_verify_split_plan.hpp,src/attention.cpp,src/verify_attn_profile.hpp,libgrimoire_gemm.so,libgrimoire_attn.so} > split-candidate.sha256
