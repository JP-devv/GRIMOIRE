#!/bin/bash
# 11-17: d7931f9 (I4A8 default + SwiGLU post-ops), host binaries of pp-1117a in the v1.10.7 image, gpu0 only.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117a; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "FIX_BUILD_PASS" logs/copy-buildpp1117a.out 2>/dev/null; do
  docker ps --format "{{.Names}}" | grep -q grim-pp1117a-builder || { echo "BUILD FAILED"; tail -30 logs/copy-buildpp1117a.out; exit 1; }
  sleep 20; done
echo "BUILD_OK $(date +%T)"
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 --mode one "$@" || echo "RUN_FAILED $*"; }
BA="--pp 4096 8192 --tg 64 --concurrency 1 --runs 3 --no-cache"
run --tag pp1117b-new --client benchy --ctx 16384 --env GRIMOIRE_I4A8_CHECK=1 --env GRIMOIRE_ONEDNN_VERBOSE=1 --benchy-args="$BA"
run --tag pp1117b-nopost --client benchy --ctx 16384 --env GRIMOIRE_I4A8_SWIGLU=0 --benchy-args="$BA"
run --tag pp1117b-copy --client copy --ctx 32768 --copy-args="--users 1,4 --reps 1 --max-tokens 1024"
echo CHAIN_PP1117B_DONE
