#!/bin/bash
# CPU-only AOT RELEASE build of 3029d5a (fast-pp-1117: I4A8 default, SwiGLU post-ops, prompt head at the end, batched
# admission with the cache, one-slot step path, QSA attention skips unselected blocks) for BOTH dies: Arc Pro B70 (G31)
# AND Arc Pro B60 / Arc B580 (G21).  Source tar grimoire-rel1117-source.tar. CLI + server + kernel libraries.
# A syntax-only pass of the changed files first, so a typo fails in minutes, not after an hour of device code.
# No GPU devices, no network, low priority, one --rm container.
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-runs/rel-1117
exec >"$B/logs/rel-build1117.out" 2>&1
exec 6>"$B/logs/rel-build1117.lock"
flock -n 6 || exit 9
test ! -e "$X"
mkdir -p "$X/bin"
tar -xf "$B/grimoire-rel1117-source.tar" -C "$X"
printf '%s\n' 3029d5a > "$X/BUILD_SOURCE"
printf '%s\n' 3029d5a83975f63213453aabb364543cd36f8b1e > "$X/BUILD_SOURCE_FULL"
date -Iseconds
nice -n 19 docker run --rm --name grim-rel1117-builder --network none --cpuset-cpus 2-13 --cpu-shares 128 \
  -e GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21 -e GRIM_ONEDNN=/onednn \
  -v "$X:/work" -v /mnt/storage/isos/onednn/install-3.13.4:/onednn:ro -w /work grimoire-dev:latest \
  bash -c 'test ! -e /dev/dri || exit 2
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    for f in src/gemm_fast.cpp src/attention.cpp src/ops.cpp src/onednn_gemm.cpp src/grimoire.cpp; do
      icpx -fsycl -fsyntax-only -std=c++20 -I include -I src -DGRIMOIRE_WITH_ONEDNN -I /onednn/include $f || exit 3; done
    echo SYNTAX_OK
    bash tools/build_grimoire_only_b70.sh /work && bash tools/build_server_only_b70.sh /work &&
    sha256sum bin/grimoire bin/grimoire-server bin/libgrimoire_gemm.so bin/libgrimoire_attn.so'
echo REL_BUILD_PASS
date -Iseconds
