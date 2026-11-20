#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
work=$PWD
base=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP
verifier=${work%/*}/verifier
stage=inputs
exec > build.log 2>&1
trap 'rc=$?; sha256sum -c "$base/serving.sha256" || rc=1; printf "stage=%s exit=%d\n" "$stage" "$rc" > build.receipt; printf "%d\n" "$rc" > build.exit; exit "$rc"' EXIT
[[ $(< "$verifier/build.exit") == 0 ]]
(cd "${work%/*}/repo" && sha256sum -c "$verifier/candidate.sha256")
sha256sum -c "$base/serving.sha256"
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force
cp -a --reflink=auto "$verifier"/libgrimoire_{gemm,attn}.so .
objects=()
for obj in "$base"/obj/*.o; do
    case ${obj##*/} in grimoire.cpp.o|attention.cpp.o) ;; *) objects+=("$obj");; esac
done
sha256sum "${objects[@]}" > base-objects.sha256
flags=(-fsycl -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -O2 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -Iinclude -Isrc -Ixe2)
stage=attention
icpx "${flags[@]}" -c src/attention.cpp -o attention.cpp.o
stage=verifier
icpx "${flags[@]}" -c verifier-profile.cpp -o grimoire.cpp.o
stage=link
icpx "${flags[@]}" -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -fsycl-max-parallel-link-jobs=4 grimoire.cpp.o attention.cpp.o "${objects[@]}" -pthread -L. \
    -lgrimoire_gemm -lgrimoire_attn '-Wl,-rpath,$ORIGIN' -o qfn-verify-attn-profile
stage=verify
strings qfn-verify-attn-profile | awk '/Verify attention: backend=block/{b=1} /Verify attention: backend=batched/{s=1} END {exit !(b&&s)}'
sha256sum "$work"/{qfn-verify-attn-profile,verifier-profile.cpp,src/attention.cpp,src/verify_attn_profile.hpp,libgrimoire_gemm.so,libgrimoire_attn.so} > candidate.sha256
