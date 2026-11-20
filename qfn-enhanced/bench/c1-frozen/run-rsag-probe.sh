#!/usr/bin/env bash
set -euo pipefail
: "${APPROVED_DEADLINE_EPOCH:?}" "${TMUX:?}"
B=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd "$B/../../.." && pwd)
S=$ROOT/receipts/prefill-rsag-build
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
D=$(mktemp -d "$ROOT/receipts/rsag-probe.XXXXXXXX")
echo "$D"
[[ $(< "$S/build-rsag.exit") == 0 ]]
(cd "$S"; sha256sum -c rsag.sha256)
sha256sum -c "$SERVING"
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
idle() {
    local rc=0 occupancy node
    for node in "${nodes[@]}"; do [[ -c $node ]] || return 1; done
    occupancy=$(sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
    [[ $rc == 1 && -z $occupancy ]]
}
idle
cursor=$(sudo -n journalctl -q -k -n 1 --no-pager --show-cursor | awk '/^-- cursor: / {print $3}')
[[ -n $cursor ]]
pids=()
socket_dir=$(mktemp -d /tmp/qfn-rsag.XXXXXXXX)
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
unset LD_LIBRARY_PATH LD_PRELOAD ZE_AFFINITY_MASK SYCL_DEVICE_FILTER SYCL_UR_TRACE SYCL_PI_TRACE
for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_TP_SHM=0 GRIMOIRE_QFN_TP4_PEER_REGRESSION=1
export GRIMOIRE_QFN_PREFILL_RSAG=1 GRIMOIRE_P2P_CHUNK_MB=1
cap=$((APPROVED_DEADLINE_EPOCH - $(date +%s) - 15))
(( cap >= 45 ))
for rank in 3 2 1 0; do
    GRIMOIRE_TP_RANK=$rank GRIMOIRE_TP_SOCKET=$socket_dir/p \
        timeout --signal=TERM --kill-after=5s "$cap" "$S/qfn-combined" \
        -m /home/haxor/models/qwen38-w4a16-snapshot --proj bf16 --ctx 32768 \
        -p transport -n 1 > "$D/r$rank.out" 2> "$D/r$rank.log" &
    pids[rank]=$!
done
for rank in 0 1 2 3; do wait "${pids[rank]}"; unset 'pids[rank]'; done
for rank in 0 1 2 3; do
    for call in 0 5; do
        grep -E "^TP4 RSAG rank=$rank call=$call peer_bytes=31457280 .* exact$" "$D/r$rank.out"
    done
done > "$D/transport.txt"
