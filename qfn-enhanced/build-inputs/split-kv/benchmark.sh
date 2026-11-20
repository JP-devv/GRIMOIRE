#!/usr/bin/env bash
# Separate frozen four-rank MTP3 profiles and unprofiled split-KV trials.
set -eo pipefail
cd -- "$(dirname -- "$0")"
work=$PWD
out=$(realpath -e -- "${1:?empty output directory}")
deadline=${2:?absolute end epoch of the newly authorized GPU window}
mode=${3:-profile}
[[ $mode == profile || $mode == split-ab || $mode == split-profile ]]
bin=$work/qfn-verify-attn-profile
manifest=candidate.sha256
if [[ $mode == split-* ]]; then bin=$work/qfn-verify-split; manifest=split-candidate.sha256; fi
[[ -d $out && -z $(ls -A -- "$out") && $deadline =~ ^[1-9][0-9]{9}$ && -n ${TMUX:-} ]]
root=/home/haxor/AI/GRIMOIRE-qfn-qgate
base=$root/diagnostics/act-xe2-tilecut-ab-6IU2nP
prior=${work%/*}/verifier/gate-ZwV33FAa
prompt_file=$root/diagnostics/act-moe-fU6ZtS/prompt.ids
frozen=$root/diagnostics/act-xmx-ab-N8CMZj/grouped-gate
model=/home/haxor/models/qwen38-w4a16-snapshot
ple=/home/haxor/models/qwen38-w4a16
draft=/home/haxor/models/qwen38-w4a16-mtp-snapshot
stage=preflight
pids=()
socket_dir=
finish() {
    local rc=$?
    trap - EXIT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done
    for pid in "${pids[@]}"; do wait "$pid" || :; done
    if [[ -n $socket_dir ]]; then rm -f -- "$socket_dir/p"; rmdir -- "$socket_dir" || rc=1; fi
    sha256sum -c "$base/serving.sha256" > "$out/serving-after.sha256" || rc=1
    printf 'stage=%s exit=%d\n' "$stage" "$rc" > "$out/receipt"
    exit "$rc"
}
exec > "$out/run.log" 2>&1
trap finish EXIT
trap 'exit 143' TERM INT
[[ $(< build.exit) == 0 && $(< build.receipt) == 'stage=verify exit=0' ]]
if [[ $mode == split-* ]]; then [[ $(< split-build.receipt) == 'stage=verify exit=0' ]]; fi
[[ $(< "$prior/receipt") == 'stage=verify exit=0' ]]
sha256sum -c "$manifest"
sha256sum -c base-objects.sha256
sha256sum -c "$base/serving.sha256"
sha256sum -c "$base/inputs.sha256"
sha256sum -c "$prior/inputs.sha256"
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null
set -u
stage=lock
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
render=(/dev/dri/renderD*)
[[ ${#render[@]} == 4 ]]
for node in "${nodes[@]}"; do [[ -c $node ]]; done
rc=0
occupancy=$(sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
[[ $rc == 1 && -z $occupancy ]] || { printf 'GPU occupancy check refused: %d %s\n' "$rc" "$occupancy"; exit 1; }
for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
unset ZE_AFFINITY_MASK UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS SYCL_UR_TRACE SYCL_PI_TRACE ZE_ENABLE_TRACING_LAYER UR_ENABLE_LAYERS
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
export GRIMOIRE_P2P=1 GRIMOIRE_P2P_CHUNK_MB=1 GRIMOIRE_QFN_TP_SYNC=fanout
export GRIMOIRE_QFN_PLE_SOURCE=$ple GRIMOIRE_PLE_RAM=0 GRIMOIRE_MTP=0 GRIMOIRE_PREFIX_CACHE=0
export GRIMOIRE_PREFILL_CHUNK=2048 GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_DEVICE_ROUTES=1
export GRIMOIRE_QFN_GROUPED_K_PREFETCH=0 GRIMOIRE_QFN_GROUPED_LIVE_TILES=1
export GRIMOIRE_QFN_XE2_GATE_UP=fp32 GRIMOIRE_QFN_XE2_DOWN=1
export GRIMOIRE_QFN_HC_DOWN_TILED=1 GRIMOIRE_QFN_HC_FUSED_PREFILL=1 GRIMOIRE_DECODE_GRAPH=0
export GRIMOIRE_PROMPT_IDS="$(< "$prompt_file")" GRIMOIRE_PRINT_IDS=1
export GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=mtp3 GRIMOIRE_QFN_MTP_SOURCE=$draft GRIMOIRE_SPEC_STATS=1
run_arm() {
    local arm=$1 kind=$2 split=$3 dir rank rc=0 status pid stem steps cursor remaining cap
    dir=$out/$arm
    local -a extra=(GRIMOIRE_QFN_VERIFY_SPLIT_KV="$split")
    case $kind in
        profile) extra+=(GRIMOIRE_VERIFY_ATTN_PROFILE=1 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_QFN_VERIFY_PROFILE=1
                         GRIMOIRE_Q4_TIMING=1 GRIMOIRE_HC_TIMING=1) ;;
        gate) extra+=(GRIMOIRE_QFN_VERIFY_ATTN_TRACE=1) ;;
        timed) extra+=(GRIMOIRE_GENERATION_PHASES=1) ;;
    esac
    mkdir "$dir"
    socket_dir=$(mktemp -d /tmp/qfn-vattn.XXXXXXXX)
    cursor=$(sudo -n journalctl -q -k -n 1 --no-pager --show-cursor | awk '/^-- cursor: / {print $3}')
    [[ -n $cursor ]]
    stage=$arm-deadline
    remaining=$((deadline - $(date +%s) - 15))
    (( remaining >= 90 )) || { echo "Not enough authorized time for $arm"; exit 124; }
    cap=$((remaining < 165 ? remaining : 165))
    stage=$arm-ranks
    for rank in 3 2 1 0; do
        env "${extra[@]}" GRIMOIRE_TP_RANK="$rank" GRIMOIRE_TP_SOCKET="$socket_dir/p" \
            timeout --signal=TERM --kill-after=4s "$cap" "$bin" -m "$model" --proj bf16 --ctx 32768 -p frozen -n 32 \
            > "$dir/r$rank.out" 2> "$dir/r$rank.log" &
        pids[rank]=$!
    done
    for rank in 0 1 2 3; do
        status=0
        wait "${pids[rank]}" || status=$?
        printf '%s rank=%d exit=%d\n' "$arm" "$rank" "$status"
        if (( status != 0 && rc == 0 )); then
            rc=$status
            for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done
        fi
    done
    pids=()
    rm -f -- "$socket_dir/p"
    rmdir -- "$socket_dir"
    socket_dir=
    sudo -n journalctl -q -k --no-pager --after-cursor "$cursor" -o cat > "$dir/kernel.log"
    awk '/xe .*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$dir/kernel.log" > "$dir/xe-fault.log"
    [[ ! -s $dir/xe-fault.log ]]
    ((rc == 0)) || exit "$rc"
    stage=$arm-parity
    for rank in 0 1 2 3; do
        stem=$dir/r$rank
        grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$stem.out"
        grep -Fxq '  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$stem.out"
        grep -Fxq "prompt ids (2706): $GRIMOIRE_PROMPT_IDS" "$stem.log"
        grep '^greedy ids (32): ' "$stem.log" > "$stem.ids"
        diff -u "$frozen/matched-r$rank.ids" "$stem.ids" > "$stem.ids.diff"
        steps=$(awk '/  spec: MTP depth 3 .* over [1-9][0-9]* steps$/ { print $(NF-1) }' "$stem.out")
        [[ $steps =~ ^[1-9][0-9]*$ ]]
        case $kind in
            profile)
                awk -v rank="$rank" -v is_split="$split" -f check-profile.awk "$stem.log" > "$stem.attention.tsv"
                [[ $(wc -l < "$stem.attention.tsv") == "$steps" ]]
                cut -f2 "$stem.attention.tsv" > "$stem.contexts"
                cmp "$dir/r0.contexts" "$stem.contexts"
                awk '1' "$stem.attention.tsv" >> "$out/attention.pending.tsv" ;;
            gate)
                awk -v rank="$rank" -v splits="$((split ? 32 : 1))" -v steps="$steps" -f check-dispatch.awk "$stem.log" > "$stem.contexts"
                cmp "$dir/r0.contexts" "$stem.contexts" ;;
            timed)
                awk -f check-timing.awk "$stem.log" > "$stem.times" ;;
        esac
    done
}
if [[ $mode == split-ab ]]; then
    run_arm gate-control gate 0
    run_arm gate-split gate 1
    printf 'arm\tprefill_ms\tdecode_ms\tprefill_tok_s\tdecode_tok_s\n' > "$out/timing.pending.tsv"
    for arm in 1-control 1-split 2-split 2-control; do
        split=0; if [[ $arm == *-split ]]; then split=1; fi
        run_arm "$arm" timed "$split"
        awk -v arm="$arm" '
            {if ($1>pp) pp=$1; if ($2>tg) tg=$2}
            END {if(NR!=4 || pp<=0 || tg<=0) exit 1; printf "%s\t%.3f\t%.3f\t%.2f\t%.2f\n",arm,pp,tg,2706000/pp,31000/tg}
        ' "$out/$arm"/r?.times >> "$out/timing.pending.tsv"
    done
    product=timing
else
    split=0; columns=fused_device_ms
    if [[ $mode == split-profile ]]; then split=1; columns=$'split_kv_device_ms\treduction_device_ms'; fi
    printf 'rank\tcontext\tlayers\t%s\tfull_attention_fenced_ms\tfenced_minus_device_ms\n' "$columns" > "$out/attention.pending.tsv"
    run_arm mtp-on-profile profile "$split"
    product=attention
fi
stage=verify
sha256sum -c "$manifest"
sha256sum -c "$base/inputs.sha256"
sha256sum -c "$prior/inputs.sha256"
sha256sum -c "$base/serving.sha256"
mv "$out/$product.pending.tsv" "$out/$product.tsv"
