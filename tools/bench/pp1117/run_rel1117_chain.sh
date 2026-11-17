#!/bin/bash
# 2026-11-17 v1.10.8-rc1 = 3029d5a (fast-pp-1117). After the Flash-Next MTP test: release build G31 + G21, image FROM v1.10.7,
# model pass (cache on) against v1.10.7 (regress-1116-2133), then served checks with the IMAGE's own binaries.
# No upload here.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-runs
T=$R/rel-1117
IMG=grimoire-b70:v1.10.8-rc1
H=/mnt/storage/Models/grimoire-ple/flash-next.hits
cd $B
until grep -q "CHAIN_FNMTP1117_DONE" logs/fnmtp1117.out 2>/dev/null || ! pgrep -f run_fnmtp1117.sh >/dev/null; do sleep 30; done
until [ -z "$(docker ps --format '{{.Names}}' | grep -E '^s1114-|grim-')" ]; do sleep 10; done
echo "REL_CHAIN_START $(date +%T)"
./build_rel1117.sh
grep -q REL_BUILD_PASS logs/rel-build1117.out || { echo "REL BUILD FAILED"; tail -30 logs/rel-build1117.out; exit 1; }
echo "STEP build PASS $(date +%T)"
python3 - "$T" <<'PY' || { echo "TARGETS FAILED"; exit 1; }
import re, sys
for f in ("grimoire", "grimoire-server", "libgrimoire_gemm.so", "libgrimoire_attn.so"):
    t = set(re.findall(rb"intel_gpu_bmg_g[0-9]+", open(sys.argv[1] + "/bin/" + f, "rb").read()))
    print(f, sorted(x.decode() for x in t))
    assert {b"intel_gpu_bmg_g31", b"intel_gpu_bmg_g21"} <= t, f
PY
echo "STEP targets PASS"
D=$R/image-1117rel
rm -rf $D; mkdir -p $D
cp $T/bin/grimoire $T/bin/grimoire-server $T/bin/libgrimoire_gemm.so $T/bin/libgrimoire_attn.so $D/
cp $T/docker-entrypoint.sh $D/
FULL=$(cat $T/BUILD_SOURCE_FULL)
cat > $D/Dockerfile <<DF
FROM grimoire-b70:v1.10.7
COPY --chmod=0755 grimoire grimoire-server libgrimoire_gemm.so libgrimoire_attn.so /grimoire/bin/
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
LABEL org.opencontainers.image.version=v1.10.8 \\
      org.opencontainers.image.revision=$FULL \\
      org.grimoire.branch=fast-pp-1117 \\
      org.grimoire.targets=intel_gpu_bmg_g31,intel_gpu_bmg_g21
DF
docker build -q -t $IMG $D || { echo "IMAGE BUILD FAILED"; exit 1; }
echo "STEP image PASS"
[ -e "$H.pre-1117" ] || cp -p "$H" "$H.pre-1117"
trap 'cp -p "$H.pre-1117" "$H"' EXIT
cp "$H.det-1105" "$H"
( cd /mnt/storage/isos/grimoire-fuse
  exec 9>"$R/gpu0.lock"; flock -n 9 || { echo "gpu0.lock held"; exit 75; }
  GRIM_IMAGE=$IMG HOSTBIN=0 bash $B/regress_all_g0_cont1116.sh 9>&- ) > $B/logs/rel1117-regress.out 2>&1
grep -q "STOP: gpu0 guard" $B/logs/rel1117-regress.out && { echo "MODEL PASS STOPPED BY GUARD"; exit 1; }
RD=$(grep -h "^results: " $B/logs/rel1117-regress.out | cut -d" " -f2)
python3 $B/cmp_regress1116.py $RD $R/regress-1116-2133 > $B/logs/rel1117-cmp.out
echo "STEP model-pass DONE $RD: vs v1.10.7 $(tail -1 $B/logs/rel1117-cmp.out)"
cp -p "$H.pre-1117" "$H"
run(){ python3 -u run_rel1116.py --tree $T --image $IMG --hostbin 0 "$@" || { echo "RUN_FAILED $*"; }; }
DQ="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag rel1117-dual-copy $DQ --client copy --copy-args="--users 1 --reps 1 --max-tokens 1024"
run --tag rel1117-dual-pfx $DQ --client pfx --copy-args="--chars 120000 --warm"
run --tag rel1117-one-copy --mode one --ctx 32768 --client copy --copy-args="--users 1,4 --reps 1 --max-tokens 1024"
run --tag rel1117-one-pfx --mode one --client pfx --ctx 70000 --env GRIMOIRE_SEQ_SLOTS=4 --copy-args="--chars 120000 --warm"
run --tag rel1117-one-c8 --mode one --ctx 16384 --client benchy --benchy-args="--pp 512 4096 --tg 128 --concurrency 1 4 8 --runs 2 --no-cache"
cp /mnt/storage/Models/grimoire-ple/flash-next.hits.frozen-1028 /mnt/models/grimoire-ple/flash-next.hits.rel1117
FN="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int8 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.rel1117"
run --tag rel1117-fn-copy $FN --client copy --copy-args="--users 1 --reps 1 --max-tokens 256"
echo "STEP validation DONE $(date +%T)"
# THE TABLE (Ian 11-17: every model, 4-bit and 8-bit, one card and two, every context, reading + writing): the rc image
L5="--pp 4096 16384 32768 65536 131072 --tg 128 --concurrency 1 --runs 1 --no-cache"
L3="--pp 4096 16384 32768 --tg 128 --concurrency 1 --runs 1 --no-cache"
LF="--pp 4096 16384 32768 60000 --tg 128 --concurrency 1 --runs 1 --no-cache"
run --tag tab1117-q4-one --mode one --ctx 140000 --env GRIMOIRE_SEQ_SLOTS=1 --client benchy --benchy-args="$L5"
run --tag tab1117-q4-tp --mode tp --proj int4 --ctx 140000 --env GRIMOIRE_SEQ_SLOTS=2 --client benchy --benchy-args="$L5"
DQ2="--mode tp --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 262144 --profile-env --env GRIMOIRE_MTP=1
    --env GRIMOIRE_SEQ_SLOTS=1 --env GRIMOIRE_MTP_K=6 --env GRIMOIRE_MTP_DRAFT_VOCAB=65536"
run --tag tab1117-q8-tp $DQ2 --client benchy --benchy-args="$L5"
run --tag tab1117-q8-one --mode one --model /models2/Qwen3.8-27B-Uncensored-FP8 --proj int8 --ctx 40000 --env GRIMOIRE_SEQ_SLOTS=1 --client benchy --benchy-args="$L3"
run --tag tab1117-fn8-tp $FN --client benchy --benchy-args="$LF"
FN4="--mode tp --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int4 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.rel1117"
run --tag tab1117-fn4-tp $FN4 --client benchy --benchy-args="$LF"
FN41="--mode one --model /models2/Qwen3.8-Flash-Next-NVFP4 --proj int4 --ctx 65536 --profile-env --env GRIMOIRE_SEQ_SLOTS=1
    --env GRIMOIRE_MTP=0 --env GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 --env GRIMOIRE_PLE_FILE=/models2/grimoire-ple/flash-next-ple.bin
    --env GRIMOIRE_EXPERT_HITS=/models2/grimoire-ple/flash-next.hits.rel1117"
run --tag tab1117-fn4-one $FN41 --client benchy --benchy-args="$L3"
echo CHAIN_REL1117_DONE
