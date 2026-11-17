#!/bin/bash
# 11-17: 38b71d8 -- batched admission with the prefix cache on (one card, 8 users of short prompts) vs the v1.10.7 rule.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "CHAIN_PP1117E_DONE\|NO BUILD" logs/pp1117e.out 2>/dev/null; do sleep 20; done
grep -q "NO BUILD" logs/pp1117e.out && exit 1
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
BA="--pp 128 512 --tg 128 --concurrency 1 4 8 --runs 2 --no-cache"
run --tag pp1117f-one-c8 --mode one --ctx 16384 --client benchy --benchy-args="$BA"
run --tag pp1117f-one-c8-old --mode one --ctx 16384 --env GRIMOIRE_ADMIT_BATCH_CACHED=0 --client benchy --benchy-args="$BA"
echo CHAIN_PP1117F_DONE
