#!/bin/bash
# 11-17: Flash-Next 4K rows of the table again WARM: the first request after a Flash-Next start reads PLE rows from the
# file while the table is copied to RAM (the sweep 4K row: 457 tok/s, tg 24). A 512-token prompt first absorbs it.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/rel-1117; IMG=grimoire-b70:v1.10.8-rc1
cd $B || exit 9
until grep -q "CHAIN_REL1117_DONE" logs/rel1117-chain.out 2>/dev/null || ! pgrep -f run_rel1117_chain.sh >/dev/null; do sleep 30; done
grep -q "CHAIN_REL1117_DONE" logs/rel1117-chain.out || { echo "REL CHAIN DID NOT FINISH"; exit 1; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -E "^s1114-|grim-")" ]; do sleep 10; done
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 0 "$@" || echo "RUN_FAILED $*"; }
for p in 8 4; do
  FN="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int$p --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
      --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
      --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.rel1117"
  run --tag tabw1117-fn$p-tp $FN --client benchy --benchy-args="--pp 512 4096 --tg 128 --concurrency 1 --runs 2 --no-cache"
done
echo CHAIN_TABFIX1117_DONE
