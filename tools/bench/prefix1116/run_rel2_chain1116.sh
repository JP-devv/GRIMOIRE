#!/bin/bash
# 2026-11-16 v1.10.7-rc2 = SRC (rc1 + fixes found by its model pass).  Same plan as run_rel_chain1116.sh, plus the
# model pass made comparable with v1.10.6's own (regress-1106-2107: hits snapshot det-1105, gpu0.lock), and the
# cache-on vs cache-off pass compared file by file.  Everything runs the IMAGE's own binaries.  No upload here.
set -u
SRC=${SRC:?}
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-runs
T=$R/rel-1116b
IMG=grimoire-b70:v1.10.7-rc2
H=/mnt/storage/Models/grimoire-ple/flash-next.hits
cd $B
./build_rel1116b.sh
grep -q REL_BUILD_PASS logs/rel-build1116b.out || { echo "REL BUILD FAILED"; exit 1; }
echo "STEP build PASS"
python3 - "$T" <<'EOF' || { echo "TARGETS FAILED"; exit 1; }
import re, sys
for f in ('grimoire', 'grimoire-server', 'libgrimoire_gemm.so', 'libgrimoire_attn.so'):
    t = set(re.findall(rb'intel_gpu_bmg_g[0-9]+', open(sys.argv[1] + '/bin/' + f, 'rb').read()))
    print(f, sorted(x.decode() for x in t))
    assert {b'intel_gpu_bmg_g31', b'intel_gpu_bmg_g21'} <= t, f
EOF
echo "STEP targets PASS"
D=$R/image-1116rel2
rm -rf $D; mkdir -p $D
cp $T/bin/grimoire $T/bin/grimoire-server $T/bin/libgrimoire_gemm.so $T/bin/libgrimoire_attn.so $D/
cp $R/image-1113rel/libdnnl.so.3.13 $D/
cp $T/docker-entrypoint.sh $D/
FULL=$(cat $T/BUILD_SOURCE_FULL)
cat > $D/Dockerfile <<EOF
FROM grimoire-b70:v1.10.6
COPY --chmod=0755 grimoire grimoire-server libgrimoire_gemm.so libgrimoire_attn.so /grimoire/bin/
COPY libdnnl.so.3.13 /grimoire/bin/libdnnl.so.3
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
LABEL org.opencontainers.image.version=v1.10.7 \\
      org.opencontainers.image.revision=$FULL \\
      org.grimoire.branch=longctx-1115 \\
      org.grimoire.targets=intel_gpu_bmg_g31,intel_gpu_bmg_g21
EOF
docker build -q -t $IMG $D || { echo "IMAGE BUILD FAILED"; exit 1; }
docker image inspect $IMG --format "image {{.Id}} size {{.Size}}"
echo "STEP image PASS"
# model pass x2 (cache on = default; then off), Flash-Next hits snapshot as v1.10.6's pass
[ -e "$H.pre-1116" ] || cp -p "$H" "$H.pre-1116"
trap 'cp -p "$H.pre-1116" "$H"' EXIT
for MODE in on off; do
  cp "$H.det-1105" "$H"
  ( cd /mnt/storage/isos/grimoire-fuse
    exec 9>"$R/gpu0.lock"; flock -n 9 || { echo "gpu0.lock held"; exit 75; }
    if [ $MODE = off ]; then export SUITE_ENV=GRIMOIRE_PREFIX_CACHE=0; fi
    GRIM_IMAGE=$IMG HOSTBIN=0 bash $B/regress_all_g0_cont1116.sh 9>&- ) > $B/logs/rel2-regress-$MODE.out 2>&1
  grep -q "STOP:" $B/logs/rel2-regress-$MODE.out && { echo "MODEL PASS ($MODE) STOPPED BY GUARD"; exit 1; }
  RD=$(grep -h '^results: ' $B/logs/rel2-regress-$MODE.out | cut -d' ' -f2)
  python3 $B/cmp_regress1116.py $RD $R/regress-1106-2107 > $B/logs/rel2-cmp-$MODE-v1106.out
  # a run whose model folder is gone (Ornith / Muse / Agnes left /mnt/storage/Models on 11-09) is not a failure
  grep -q "FAILED=" <(tail -1 $B/logs/rel2-cmp-$MODE-v1106.out) && { echo "MODEL PASS ($MODE) HAS FAILED RUNS"; grep "^FAILED" $B/logs/rel2-cmp-$MODE-v1106.out; exit 1; }
  echo "STEP model-pass-$MODE DONE $RD: vs v1.10.6 $(tail -1 $B/logs/rel2-cmp-$MODE-v1106.out)"
done
cp -p "$H.pre-1116" "$H"
python3 $B/cmp_regress1116.py $(grep -h '^results: ' $B/logs/rel2-regress-on.out | cut -d' ' -f2) \
    $(grep -h '^results: ' $B/logs/rel2-regress-off.out | cut -d' ' -f2) > $B/logs/rel2-cmp-onoff.out
echo "STEP compare: cache on vs off $(tail -1 $B/logs/rel2-cmp-onoff.out)"
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 0 "$@" || { echo "RUN_FAILED $*"; exit 1; }; }
run --tag rel2-one-pfx --mode one --client pfx --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --copy-args="--chars 120000 --warm"
run --tag rel2-one-copy --mode one --client copy --ctx 32768 --copy-args="--users 1,4 --reps 1 --max-tokens 1024"
run --tag rel2-tp-probe --mode tp --proj int4 --client probe --ctx 136000 --env GRIMOIRE_SPEC_TIMING=1
run --tag rel2-tp-pfx --mode tp --proj int4 --client pfx --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --copy-args="--chars 120000 --warm"
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag rel2-dualq-pfx $DQ --client pfx --copy-args="--chars 120000 --warm"
run --tag rel2-dualq-benchy $DQ --client benchy --benchy-args="--pp 128 2048 4096 --tg 32 256 --runs 3"
# Ian's DUAL-FLASH-NEXT profile (the prefix cache refuses Qwen4-Exp: it must still start and answer); a hits COPY
cp /mnt/storage/Models/grimoire-ple/flash-next.hits.frozen-1028 /mnt/models/grimoire-ple/flash-next.hits.rel1116
FN="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int8 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.rel1116"
run --tag rel2-fn-pfx $FN --client pfx --copy-args="--chars 120000 --warm"
echo CHAIN_REL2_DONE
