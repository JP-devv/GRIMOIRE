#!/bin/bash
# 2026-11-15 x (55a9eea): copy depth 63 default + 64-row verify with row slots 32, verify-only 33..64-row INT4,
# VRAM back.  ONE B70: 1 user short (texts vs u-short), 4/8 users short (vs w-short48), 61K 1 + 4 users (TTFT and
# prefill chunk vs u-61k), TWO B70s Ian's INT8 TP short 1/4/8 users (vs w-tp-short).
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-runs
cd $B
until grep -q CHAIN_Z_DONE logs/long-chain1115z.out 2>/dev/null; do sleep 10; done
./build_copy1115x.sh
grep -q FIX_BUILD_PASS logs/copy-build1115x.out || { echo "build x not passed"; exit 1; }
echo BUILD_X_PASS
T=$R/copy-1115x
U=/models2/Qwen3.8-27B-Uncensored-FP8
run(){ python3 -u run_copy1114.py "$@" || echo "RUN_FAILED $*"; }
run --tree $T --tag x-short1 --mode one --client copy --env GRIMOIRE_SPEC_TIMING=1 --copy-args="--users 1 --reps 1 --max-tokens 512"
run --tree $T --tag x-short48 --mode one --client copy --env GRIMOIRE_SPEC_TIMING=1 --copy-args="--users 4,8 --reps 1 --max-tokens 1024"
run --tree $T --tag x-61k --mode one --client copy --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --env GRIMOIRE_SPEC_TIMING=1 \
    --copy-args="--users 1,4 --reps 1 --max-tokens 1024 --long-chars 245000"
run --tree $T --tag x-tp-short --mode tp --model $U --proj int8 --client copy --env GRIMOIRE_SPEC_TIMING=1 --copy-args="--users 1,4,8 --reps 1 --max-tokens 1024"
echo CHAIN_X_DONE
