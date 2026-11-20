#!/usr/bin/env bash
set -eo pipefail
bin=$(realpath -- "${1:?isolated JIT binary}")
out=$(realpath -- "${2:?fresh output directory}")
[[ -x $bin && -d $out && -z $(ls -A -- "$out") ]]
job=${out##*/}
stage=setup
sockets=
exec > "$out/run.log" 2>&1
trap 'rc=$?; if [[ -n $sockets ]]; then rm -f -- "$sockets/p"; rmdir -- "$sockets"; fi; printf "stage=%s exit=%s\n" "$stage" "$rc" > "$out/receipt.tmp"; mv "$out/receipt.tmp" "$out/receipt"; if [[ -n ${TMUX:-} ]]; then tmux wait-for -S "$job-done"; fi' EXIT
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null
set -u
stage=lock
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -w "${QFN_LOCK_WAIT:-60}" 9
render=(/dev/dri/renderD*)
[[ ${#render[@]} == 4 && -c ${render[0]} ]]
command -v fuser >/dev/null
sudo -n true
rc=0
occupancy=$(sudo -n fuser "${render[@]}" 2>&1) || rc=$?
if [[ $rc != 1 || -n $occupancy ]]; then
    printf 'GPU occupancy check refused: exit=%d %s\n' "$rc" "$occupancy"
    exit 1
fi
stage=collective
sha256sum "$bin" > "$out/binary.sha256"
unset GRIMOIRE_PP_RANK GRIMOIRE_PP_WORLD_SIZE GRIMOIRE_DEVICES ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
export GRIMOIRE_QFN_TP4_PEER_REGRESSION=1 GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_TP_SHM=0
# Exercise a whole 5 MiB push and a smaller message reusing the same slots.
export GRIMOIRE_P2P_CHUNK_MB=${QFN_TEST_PEER_CHUNK_MB:-8}
sockets=$(mktemp -d /tmp/qfn-peer.XXXXXXXX)
export GRIMOIRE_TP_SOCKET=$sockets/p
pids=()
for rank in 0 1 2 3; do
    GRIMOIRE_TP_RANK=$rank timeout --signal=TERM "${QFN_GPU_TIMEOUT:-150s}" "$bin" -m unused -p probe -n 1 \
        > "$out/rank-$rank.log" 2>&1 &
    pids[rank]=$!
done
rc=0
for rank in 0 1 2 3; do
    if wait "${pids[rank]}"; then status=0; else status=$?; fi
    printf 'rank=%d exit=%d\n' "$rank" "$status"
    if (( status && !rc )); then rc=$status; fi
done
(( rc == 0 )) || exit "$rc"
stage=checker
for rank in 0 1 2 3; do
    for call in 0 1 2 3; do
        bytes=1495040; if (( call == 0 || call == 3 )); then bytes=5242880; fi
        grep -x "TP4 peer regression rank=$rank call=$call bytes=$bytes exact" "$out/rank-$rank.log"
    done
done
sha256sum -c "$out/binary.sha256"
