#!/bin/bash
# 11-17: Flash-Next on two cards (Ian DUAL-FLASH-NEXT profile) -- MTP off vs MTP depth 1 / 2 (it lost at depth 3 on
# 11-06: 46.9 vs 52.9 tok/s; urakozz runs depth 1-3 by acceptance: code 70 vs 49 plain). Copy bench texts, 1 user.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-runs
T=$R/pp-1117c; IMG=grimoire-b70:v1.10.7
cd $B || exit 9
until grep -q "CHAIN_SWEEP1117_DONE" logs/sweep1117.out 2>/dev/null || ! pgrep -f run_sweep1117.sh >/dev/null; do sleep 20; done
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 1 "$@" || echo "RUN_FAILED $*"; }
FN="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int8 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.sw1117"
CA="--users 1 --reps 1 --max-tokens 512"
for k in 0 1 2; do
  if [ $k = 0 ]; then E="--env GRIMOIRE_MTP=0"; else E="--env GRIMOIRE_MTP=1 --env GRIMOIRE_MTP_K=$k --env GRIMOIRE_SPEC_TIMING=1"; fi
  run --tag fn1117-k$k $FN $E --client copy --copy-args="$CA"
done
echo CHAIN_FNMTP1117_DONE
