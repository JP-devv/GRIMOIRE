set -eo pipefail
cd ~/AI/GRIMOIRE-qfn-integration/qfn-enhanced/build-inputs/combined-c1
source /opt/intel/oneapi/setvars.sh --force
icpx --version
make -f Makefile.combined -j4 TLA="$HOME/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-O5gnlh/sycl-tla" qfn-combined
sha256sum qfn-combined libgrimoire_attn.so libgrimoire_gemm.so
