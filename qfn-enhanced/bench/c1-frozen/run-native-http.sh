#!/usr/bin/env bash
set -euo pipefail
: "${APPROVED_DEADLINE_EPOCH:?}" "${BENCH_PORT:?}" "${TMUX:?}"
B=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd "$B/../../.." && pwd)
variant=${NATIVE_HTTP_VARIANT:-baseline}
context=${NATIVE_HTTP_CONTEXT:-32768}
[[ $context == 32768 || $context == 262144 ]]
case "$variant" in
    baseline) S=$ROOT/qfn-enhanced/build-inputs/combined-c1; manifest=rebuilt.sha256 ;;
    rsag) S=$ROOT/receipts/prefill-rsag-build; manifest=rsag.sha256 ;;
    causal) S=$ROOT/receipts/causal-tiles-build; manifest=causal-tiles.sha256 ;;
    prefill-combined) S=$ROOT/receipts/prefill-combined-build; manifest=prefill-combined.sha256 ;;
    hc-up) S=$ROOT/receipts/hc-up-mean-build; manifest=hc-up-mean.sha256 ;;
    *) echo "Unknown native serving variant: $variant" >&2; exit 2 ;;
esac
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
D=$(mktemp -d "$ROOT/receipts/native-http.XXXXXXXX")
echo "$D"
(cd "$S"; sha256sum -c "$manifest")
cp "$S/$manifest" "$D/build.sha256"
printf '%s\n' "$variant" > "$D/variant"
printf '%s\n' "$context" > "$D/context"
(cd "$B"; sha256sum -c inputs.sha256; python3 checkpoint-lock.py)
sha256sum -c "$SERVING"
[[ $BENCH_PORT =~ ^[0-9]{4,5}$ ]]
(( BENCH_PORT <= 65535 && BENCH_PORT != 19622 ))
[[ -z $(ss -H -ltn "sport = :$BENCH_PORT") ]]
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
idle() {
    local rc=0 occupants node
    for node in "${nodes[@]}"; do [[ -c $node ]] || return 1; done
    occupants=$(sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
    [[ $rc == 1 && -z $occupants ]]
}
idle
cursor=$(sudo -n journalctl -q -k -n 1 --no-pager --show-cursor | awk '/^-- cursor: / {print $3}')
[[ -n $cursor ]]
pids=()
socket_dir=$(mktemp -d /tmp/qfn-http.XXXXXXXX)
finish() {
    local rc=$? pid
    trap - EXIT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done
    for pid in "${pids[@]}"; do wait "$pid" || :; done
    rm -f -- "$socket_dir/p"
    rmdir -- "$socket_dir" || rc=1
    idle || rc=1
    sha256sum -c "$SERVING" > "$D/serving-after.sha256" || rc=1
    sudo -n journalctl -q -k --no-pager --after-cursor "$cursor" -o cat > "$D/kernel.log" || rc=1
    awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
    [[ ! -s $D/xe-fault.log ]] || rc=1
    echo "$rc" > "$D/exit"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
python3 -B "$B/render-fixtures.py" "$D/fixtures"
unset LD_LIBRARY_PATH LD_PRELOAD ZE_AFFINITY_MASK SYCL_DEVICE_FILTER SYCL_UR_TRACE SYCL_PI_TRACE
for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
export GRIMOIRE_P2P=1 GRIMOIRE_P2P_CHUNK_MB=1 GRIMOIRE_QFN_TP_SYNC=fanout
export GRIMOIRE_QFN_PLE_SOURCE=/home/haxor/models/qwen38-w4a16 GRIMOIRE_PLE_RAM=0
export GRIMOIRE_MTP=0 GRIMOIRE_PREFIX_CACHE=0 GRIMOIRE_DECODE_GRAPH=0 GRIMOIRE_SERVER_WARMUP=0
export GRIMOIRE_PREFILL_CHUNK=2048 GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_DEVICE_ROUTES=1
export GRIMOIRE_QFN_GROUPED_K_PREFETCH=0 GRIMOIRE_QFN_GROUPED_LIVE_TILES=1
export GRIMOIRE_QFN_XE2_GATE_UP=fp32 GRIMOIRE_QFN_XE2_DOWN=1
export GRIMOIRE_QFN_HC_DOWN_TILED=1 GRIMOIRE_QFN_HC_FUSED_PREFILL=1
if [[ $variant != baseline ]]; then export GRIMOIRE_QFN_PREFILL_RSAG=1; fi
if [[ $variant == causal || $variant == prefill-combined ]]; then export GRIMOIRE_QFN_PREFILL_CAUSAL_TILES=1; fi
if [[ $variant == hc-up || $variant == prefill-combined ]]; then export GRIMOIRE_QFN_HC_FUSED_UP=1; fi
compgen -e GRIMOIRE_ | sort | while IFS= read -r key; do printf '%s=%s\n' "$key" "${!key}"; done > "$D/options.env"
cap=$((APPROVED_DEADLINE_EPOCH - $(date +%s) - 20))
(( cap >= 120 ))
for rank in 3 2 1 0; do
    GRIMOIRE_TP_RANK=$rank GRIMOIRE_TP_SOCKET=$socket_dir/p \
        timeout --signal=TERM --kill-after=8s "$cap" "$S/qfn-server" \
        -m /home/haxor/models/qwen38-w4a16-snapshot --proj bf16 --ctx "$context" \
        --host 127.0.0.1 --port "$BENCH_PORT" > "$D/r$rank.out" 2> "$D/r$rank.log" &
    pids[rank]=$!
done
url=http://127.0.0.1:$BENCH_PORT
ready_by=$((SECONDS + cap - 30))
until curl --noproxy '*' -fs --max-time 2 "$url/health" > "$D/health"; do
    kill -0 "${pids[0]}" 2>/dev/null
    (( SECONDS < ready_by ))
    sleep 1
done
for file in "$D"/r?.out; do
    grep -Fxq '  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$file"
    grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$file"
done
timeout --kill-after=2s "$((APPROVED_DEADLINE_EPOCH - $(date +%s) - 20))" \
    python3 -B "$B/served-fixture.py" "$url/v1/completions" "$D/fixtures/frozen.request.json" \
    "$D/fixtures/frozen.expected-prefix.ids" > "$D/served.json"

if [[ $variant != baseline ]]; then
    for rank in 0 1 2 3; do
        grep -Fxq "QFN prefill RSAG rank=$rank rows=2048 calls=97" "$D/r$rank.log"
    done
fi
