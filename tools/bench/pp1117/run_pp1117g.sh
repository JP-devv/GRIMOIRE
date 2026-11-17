#!/bin/bash
# 11-17: Ian DUAL profile with GRIMOIRE_SEQ_SLOTS=2 instead of 1: with one slot the server never takes the batched
# speculative path, so copy drafting (copy/edit 462-605 tok/s on this model, README) never runs (11-17 copy test: ~150).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "CHAIN_PP1117F_DONE" logs/pp1117f.out 2>/dev/null || ! pgrep -f run_pp1117f.sh >/dev/null; do sleep 20; done
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag pp1117g-dq-s2-copy $DQ --env GRIMOIRE_SEQ_SLOTS=2 --client copy --copy-args="--users 1 --reps 1 --max-tokens 1024"
run --tag pp1117g-dq-s1-copy $DQ --env GRIMOIRE_SEQ_SLOTS=1 --client copy --copy-args="--users 1 --reps 1 --max-tokens 1024"
echo CHAIN_PP1117G_DONE
