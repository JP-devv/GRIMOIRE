#!/bin/bash
# 11-17: 3f84aea (pp-1117c) -- the prompt head at the prompt end: one card benchy (no 4,032 + 65 split) and the
# prefix test (A, A again, next turn) one card and on Ian DUAL profile, against yesterday (rel2-one-pfx / rel2-dualq-pfx).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "CHAIN_PP1117D_DONE\|BUILD2 FAILED" logs/pp1117d.out 2>/dev/null; do sleep 20; done
grep -q "BUILD2 FAILED" logs/pp1117d.out && { echo "NO BUILD"; exit 1; }
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
run --tag pp1117e-one-benchy --mode one --ctx 16384 --client benchy --benchy-args="--pp 4096 8192 --tg 64 --concurrency 1 --runs 3 --no-cache"
run --tag pp1117e-one-pfx --mode one --client pfx --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --copy-args="--chars 120000 --warm"
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag pp1117e-dq-pfx $DQ --env GRIMOIRE_I8A8=1 --client pfx --copy-args="--chars 120000 --warm"
echo CHAIN_PP1117E_DONE
