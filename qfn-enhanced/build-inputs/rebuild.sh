#!/usr/bin/env bash
set -eo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: bash rebuild.sh VARIANT NEW_BUILD_DIR SYCL_TLA_DIR" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "$0")" && pwd)
variant=$1
target=qfn-combined
inputs=$root/combined-c1
patches=()
case $variant in
    combined-c1) ;;
    prefill-cost) inputs=$root/prefill-cost; target=qfn-hc-m64 ;;
    row-replay) patches=(row-replay.patch) ;;
    hc-resident) patches=(qfn-hc-resident-pack.patch) ;;
    hc-fused) patches=(hc-fused-bf16.patch) ;;
    coalesced-v) patches=(qfn-split-kv-coalesced-v.patch) ;;
    batch-head) patches=(qfn-verify-batch-head.patch) ;;
    prefill-rsag) patches=(qfn-prefill-rsag.patch qfn-prefill-rsag-probe.patch qfn-prefill-rsag-receipt.patch) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
tla=$(realpath -- "$3")
[[ $(git -C "$tla" rev-parse HEAD) == 87f6850680a580654b9ea2c80dbc01aeb36ad231 ]]
[[ -z $(git -C "$tla" status --porcelain) ]]
mkdir -- "$2"
work=$(realpath -- "$2")
cp -a "$inputs"/{src,include,tools,xe2,regressions,source.sha256,Makefile.combined} "$work/"
cd "$work"
sha256sum -c source.sha256
for overlay in "${patches[@]}"; do
    patch -p1 < "$root/$overlay"
done
source /opt/intel/oneapi/setvars.sh --force
icpx --version
make -f Makefile.combined -j4 TLA="$tla" "$target" qfn-server
sha256sum "$target" qfn-server libgrimoire_attn.so libgrimoire_gemm.so > rebuilt.sha256
