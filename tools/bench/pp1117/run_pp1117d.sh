#!/bin/bash
# 11-17: 082c245 (pp-1117c host binaries in the v1.10.7 image) -- two cards: Ian DUAL profile (Uncensored FP8 as
# int8) with GRIMOIRE_I8A8=1 vs off, Swift INT4 TP with the I4A8 default vs off; then one card DFlash2 with the
# INT4 small-M drafter (GRIMOIRE_DFLASH_INT4_SMALLM=1) on the copy bench (new-text decode vs yesterday MTP).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "BUILD2_PASS\|BUILD2_FAIL" logs/pp1117c-queue.out 2>/dev/null; do sleep 20; done
grep -q BUILD2_PASS logs/pp1117c-queue.out || { echo "BUILD2 FAILED"; tail -30 logs/copy-buildpp1117c.out; exit 1; }
echo "BUILD2_OK $(date +%T)"
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
BA="--pp 2048 4096 16384 --tg 128 --runs 2 --no-cache"
run --tag pp1117d-dq-i8 $DQ --env GRIMOIRE_I8A8=1 --client benchy --benchy-args="$BA"
run --tag pp1117d-dq-base $DQ --client benchy --benchy-args="$BA"
run --tag pp1117d-dq-i8-copy $DQ --env GRIMOIRE_I8A8=1 --client copy --copy-args="--users 1 --reps 1 --max-tokens 512"
run --tag pp1117d-tp-i4 --mode tp --proj int4 --ctx 32768 --client benchy --benchy-args="$BA"
run --tag pp1117d-tp-i4off --mode tp --proj int4 --ctx 32768 --env GRIMOIRE_I4A8=0 --client benchy --benchy-args="$BA"
run --tag pp1117d-one-df2 --mode one --ctx 32768 --env GRIMOIRE_MTP=0 --env GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 \
    --env GRIMOIRE_DFLASH_INT4_SMALLM=1 --client copy --copy-args="--users 1 --reps 1 --max-tokens 1024"
echo CHAIN_PP1117D_DONE
