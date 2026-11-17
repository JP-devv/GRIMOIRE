#!/bin/bash
# 11-17: one B70, Swift INT4, where a 4K prompt goes (GRIMOIRE_TIME_LAYER=all) -- default vs GRIMOIRE_I4A8=1,
# then I4A8 unfenced; v1.10.7 image binaries. Comparison target: inferzero 2,772 tok/s at 4K on a B70.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/rel-1116b; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 0 --mode one --client benchy --ctx 16384 "$@" || echo "RUN_FAILED $*"; }
BA="--pp 4096 --tg 16 --concurrency 1 --runs 2 --no-cache"
run --tag pp1117-def-tl --env GRIMOIRE_TIME_LAYER=all --benchy-args="$BA"
run --tag pp1117-i4a8-tl --env GRIMOIRE_TIME_LAYER=all --env GRIMOIRE_I4A8=1 --benchy-args="$BA"
run --tag pp1117-i4a8 --env GRIMOIRE_I4A8=1 --benchy-args="--pp 4096 16384 --tg 64 --concurrency 1 --runs 3 --no-cache"
run --tag pp1117-def --benchy-args="--pp 4096 16384 --tg 64 --concurrency 1 --runs 3 --no-cache"
echo CHAIN_PP1117_DONE
