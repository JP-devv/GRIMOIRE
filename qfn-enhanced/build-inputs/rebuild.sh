#!/usr/bin/env bash
set -eo pipefail
if [[ $# != 3 || ( $1 != combined-c1 && $1 != prefill-cost ) ]]; then
    echo "Usage: bash rebuild.sh combined-c1|prefill-cost NEW_BUILD_DIR SYCL_TLA_DIR" >&2
    exit 2
fi
inputs=$(cd -- "$(dirname -- "$0")/$1" && pwd)
tla=$(realpath -- "$3")
[[ $(git -C "$tla" rev-parse HEAD) == 87f6850680a580654b9ea2c80dbc01aeb36ad231 ]]
[[ -z $(git -C "$tla" status --porcelain) ]]
mkdir -- "$2"
work=$(realpath -- "$2")
cp -a "$inputs"/{src,include,tools,xe2,regressions,source.sha256,Makefile.combined} "$work/"
cd "$work"
sha256sum -c source.sha256
source /opt/intel/oneapi/setvars.sh --force
icpx --version
target=qfn-combined
if [[ $1 == prefill-cost ]]; then
    target=qfn-hc-m64
fi
make -f Makefile.combined -j4 TLA="$tla" "$target" qfn-server
sha256sum "$target" qfn-server libgrimoire_attn.so libgrimoire_gemm.so > rebuilt.sha256
