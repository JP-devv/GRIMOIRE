#!/usr/bin/env bash
# CPU only: tests, clean C1 CLI build and two libraries, then a deployable pin.
set -eo pipefail
cd -- "$(dirname -- "$0")"
work=$PWD
staging=${work%/*}
base=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics
serving=$base/act-xe2-tilecut-ab-6IU2nP/serving.sha256
TLA=$base/act-xe2-O5gnlh/sycl-tla
stage=inputs
exec > build.log 2>&1
trap 'rc=$?; if ! sha256sum -c "$serving"; then stage=serving-check; rc=1; fi; printf "stage=%s exit=%d\n" "$stage" "$rc" > build.receipt; printf "%d\n" "$rc" > build.exit; exit "$rc"' EXIT
sha256sum -c "$serving"
[[ $(git -C "$TLA" rev-parse HEAD) == 87f6850680a580654b9ea2c80dbc01aeb36ad231 ]]
sha256sum -c source.sha256
stage=host-regressions
routes=regressions/2026-11-20-smallm-moe/test_routes.cpp
packed=regressions/2026-11-20-qfn-packed-n64/test_packed_n64.cpp
g++ -std=c++20 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc "$routes" -o test-small-moe-routes
./test-small-moe-routes
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude "$packed" -o test-packed-n64
./test-packed-n64
g++ -std=c++20 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc \
    regressions/2026-11-20-qfn-hc-m64/test_hc_schedule.cpp -o test-hc-m64
./test-hc-m64
bash "$staging/small-moe/qualification/regressions/2026-11-20-small-moe-qualification/test-runner.sh"
bash "$staging/packed-n64/qualification/regressions/2026-11-20-packed-runner/test-runner-gates.sh"
bash "$staging/packed-n64-device-parity/test-dispatch.sh"
stage=toolchain
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force
(cd "$staging/packed-n64-device-parity" && sha256sum -c fixture.sha256 && ./test-packed-n64-device --host-only)
stage=full-scratch-build
make -f Makefile.combined -j "${QFN_PREFILL_BUILD_JOBS:-4}" TLA="$TLA" qfn-hc-m64
stage=verify
strings qfn-hc-m64 | awk '
    /^GRIMOIRE_QFN_SMALL_MOE$/ {small=1} /^GRIMOIRE_QFN_GROUPED_PACKED_N64$/ {packed=1}
    /moe.small_m_live/ {sm_path=1} /QFN packed N64:/ {n64_path=1}
    /^GRIMOIRE_QFN_HC_DOWN_M64$/ {hc=1} /^hc.down.m64$/ {hc_path=1}
    END {exit !(small && packed && sm_path && n64_path && hc && hc_path)}'
readelf -d qfn-hc-m64 | awk '
    /NEEDED.*libgrimoire_gemm\.so/ {gemm=1} /NEEDED.*libgrimoire_attn\.so/ {attn=1}
    /(RUNPATH|RPATH).*\[\$ORIGIN\]/ {origin=1} END {exit !(gemm && attn && origin)}'
sha256sum "$work"/{qfn-hc-m64,libgrimoire_gemm.so,libgrimoire_attn.so} > candidate.sha256
sha256sum -c candidate.sha256
