#!/bin/bash
# 11-17 Ian: "wrapzii and his team only work with 64k context and more ... i need results for every context".
# Reading (pp) and writing (tg128 after the prompt) at 4K / 16K / 32K / 64K / 128K, build pp-1117c (38b71d8) host
# binaries in the v1.10.7 image, I4A8 from 1536 rows (= 0c0d7e3). Ian DUAL first (2 slots: copy drafting), then his
# DUAL-FLASH-NEXT profile (ctx 65536), then Swift INT4 on one card and on two.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until ! pgrep -f run_pp1117g.sh >/dev/null; do sleep 15; done
until [ -z "$(docker ps --format "{{.Names}}" | grep -E "^s1114-|grim-")" ]; do sleep 10; done
echo "SWEEP_START $(date +%T)"
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
LONG="--pp 4096 16384 32768 65536 131072 --tg 128 --concurrency 1 --runs 2 --no-cache"
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=2 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag sw1117-dual $DQ --client benchy --benchy-args="$LONG"
cp /mnt/storage/Models/grimoire-ple/flash-next.hits.frozen-1028 /mnt/models/grimoire-ple/flash-next.hits.sw1117
FN="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int8 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.sw1117"
run --tag sw1117-flash $FN --client benchy --benchy-args="--pp 4096 16384 32768 60000 --tg 128 --concurrency 1 --runs 2 --no-cache"
run --tag sw1117-one --mode one --ctx 140000 --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_I4A8_MIN_M=1536 --client benchy --benchy-args="$LONG"
run --tag sw1117-tp --mode tp --proj int4 --ctx 140000 --env GRIMOIRE_SEQ_SLOTS=2 --env GRIMOIRE_I4A8_MIN_M=1536 --client benchy --benchy-args="$LONG"
echo CHAIN_SWEEP1117_DONE
