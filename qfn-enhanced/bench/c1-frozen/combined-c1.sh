#!/usr/bin/env bash
# One native C1 request. Run only in a newly authorized exclusive GPU window.
set -eo pipefail
export LC_ALL=C
cd -- "$(dirname -- "$0")"
B=$PWD
S=/tmp/qfn-catchup.pf1/combined-c1
CHECKS=/tmp/qfn-catchup.pf1/small-moe/qualification
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
BIN=$S/qfn-combined
(( $# == 2 )) || { echo 'Usage: combined-c1.sh EMPTY_OUTPUT_DIR APPROVED_DEADLINE_EPOCH' >&2; exit 2; }
D=$(realpath -e -- "$1")
deadline=$2
[[ -d $D && -z $(ls -A -- "$D") && $deadline =~ ^[1-9][0-9]{9}$ && -n ${TMUX:-} ]]
stage=preflight
sock=
pids=()
finish() {
    local rc=$? pid
    trap - EXIT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done
    for pid in "${pids[@]}"; do wait "$pid" || :; done
    if [[ -n $sock ]]; then rm -f -- "$sock/p"; rmdir -- "$sock" || rc=1; fi
    if ! sha256sum -c "$SERVING" > "$D/serving-after.sha256"; then stage=serving-check; rc=1; fi
    printf 'stage=%s exit=%d\n' "$stage" "$rc" > "$D/receipt"
    exit "$rc"
}
exec > "$D/run.log" 2>&1
trap finish EXIT
trap 'exit 143' TERM INT
check_pins() {
    [[ $(< "$S/build.receipt") == 'stage=verify exit=0' ]]
    cmp native-combined.sha256 "$S/candidate.sha256"
    sha256sum -c native-combined.sha256
    sha256sum -c native.sha256
    sha256sum -c inputs.sha256
    sha256sum -c combined-c1.sha256
    (cd "$CHECKS"; sha256sum -c runner.sha256)
    sha256sum -c "$SERVING"
    for item in qfn-combined libgrimoire_gemm.so libgrimoire_attn.so; do
        [[ $(realpath -e -- "$S/$item") == "$S/$item" ]]
    done
}
idle() {
    local node occupancy rc=0
    local -a render=(/dev/dri/renderD*) nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
    [[ ${#render[@]} == 4 ]] || return 1
    for node in "${nodes[@]}"; do [[ -c $node ]] || return 1; done
    occupancy=$(timeout 10s sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
    [[ $rc == 1 && -z $occupancy ]] || { printf 'GPU occupancy check refused: %d %s\n' "$rc" "$occupancy"; return 1; }
}
check_pins
cp native-combined.sha256 "$D/"
python3 -B checkpoint-lock.py
unset LD_LIBRARY_PATH LD_PRELOAD
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force
set -u
ldd "$BIN" > "$D/libraries.txt"
for lib in libgrimoire_gemm.so libgrimoire_attn.so; do
    path=$(awk -v lib="$lib" '$1==lib && $2=="=>" {print $3}' "$D/libraries.txt")
    [[ $(realpath -e -- "$path") == "$S/$lib" ]]
done
stage=lock
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
idle
cursor=$(sudo -n journalctl -q -k -n 1 --no-pager --show-cursor | awk '/^-- cursor: / {print $3}')
[[ -n $cursor ]]
for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
unset ZE_AFFINITY_MASK UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS SYCL_UR_TRACE SYCL_PI_TRACE ZE_ENABLE_TRACING_LAYER UR_ENABLE_LAYERS SYCL_DEVICE_FILTER SYCL_BE
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
export GRIMOIRE_P2P=1 GRIMOIRE_P2P_CHUNK_MB=1 GRIMOIRE_QFN_TP_SYNC=fanout
export GRIMOIRE_QFN_PLE_SOURCE=/home/haxor/models/qwen38-w4a16 GRIMOIRE_PLE_RAM=0 GRIMOIRE_MTP=0 GRIMOIRE_PREFIX_CACHE=0
export GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=mtp3 GRIMOIRE_QFN_MTP_SOURCE=/home/haxor/models/qwen38-w4a16-mtp-snapshot GRIMOIRE_SPEC_STATS=1
export GRIMOIRE_PREFILL_CHUNK=2048 GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_DEVICE_ROUTES=1
export GRIMOIRE_QFN_GROUPED_K_PREFETCH=0 GRIMOIRE_QFN_GROUPED_LIVE_TILES=1
export GRIMOIRE_QFN_XE2_GATE_UP=fp32 GRIMOIRE_QFN_XE2_DOWN=1 GRIMOIRE_QFN_XE2_DECODE_M8=0
export GRIMOIRE_QFN_HC_DOWN_TILED=1 GRIMOIRE_QFN_HC_FUSED_PREFILL=1 GRIMOIRE_DECODE_GRAPH=0
export GRIMOIRE_QFN_VERIFY_SPLIT_KV=0 GRIMOIRE_QFN_VERIFY_MERGE_WEIGHTS=0
export GRIMOIRE_QFN_SMALL_MOE=1 GRIMOIRE_QFN_GROUPED_PACKED_N64=1
export GRIMOIRE_PROMPT_IDS="$(< prompt.ids)" GRIMOIRE_PRINT_IDS=1 GRIMOIRE_GENERATION_PHASES=1
compgen -e GRIMOIRE_ | sort | while IFS= read -r key; do printf '%s=%s\n' "$key" "${!key}"; done > "$D/options.env"
stage=deadline
remaining=$((deadline - $(date +%s) - 20))
(( remaining >= 90 )) || { echo 'Not enough authorized time'; exit 124; }
cap=$((remaining < 165 ? remaining : 165))
sock=$(mktemp -d /tmp/qfn-combined.XXXXXXXX)
stage=ranks
for rank in 3 2 1 0; do
    GRIMOIRE_TP_RANK=$rank GRIMOIRE_TP_SOCKET=$sock/p \
        timeout --signal=TERM --kill-after=4s "$cap" "$BIN" \
        -m /home/haxor/models/qwen38-w4a16-snapshot --proj bf16 --ctx 32768 -p frozen -n 32 \
        > "$D/r$rank.out" 2> "$D/r$rank.log" &
    pids[rank]=$!
done
for rank in 0 1 2 3; do
    status=0
    wait "${pids[rank]}" || status=$?
    unset 'pids[rank]'
    printf 'rank=%d exit=%d\n' "$rank" "$status"
    (( status == 0 )) || exit "$status"
done
stage=kernel
idle
sudo -n journalctl -q -k --no-pager --after-cursor "$cursor" -o cat > "$D/kernel.log"
awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
[[ ! -s $D/xe-fault.log ]]
check_pins
stage=parity
bash "$B/combined-c1-score.sh" "$D" > "$D/result.pending.json"
mv "$D/result.pending.json" "$D/result.json"
stage=targets
jq -e '.targets_pass' "$D/result.json" > /dev/null
stage=verify
