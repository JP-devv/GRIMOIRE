#!/bin/bash
# 2026-11-15 x2: resume of chain x after the power cut (x-short48 died at 8 users ~11:28 CET).  Build x = 55a9eea.
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-runs
cd $B
T=$R/copy-1115x
U=/models2/Qwen3.8-27B-Uncensored-FP8
run(){ python3 -u run_copy1114.py "$@" || echo "RUN_FAILED $*"; }
run --tree $T --tag x-short48b --mode one --client copy --env GRIMOIRE_SPEC_TIMING=1 --copy-args="--users 4,8 --reps 1 --max-tokens 1024"
run --tree $T --tag x-61k --mode one --client copy --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --env GRIMOIRE_SPEC_TIMING=1 \
    --copy-args="--users 1,4 --reps 1 --max-tokens 1024 --long-chars 245000"
run --tree $T --tag x-tp-short --mode tp --model $U --proj int8 --client copy --env GRIMOIRE_SPEC_TIMING=1 --copy-args="--users 1,4,8 --reps 1 --max-tokens 1024"
echo CHAIN_X2_DONE
