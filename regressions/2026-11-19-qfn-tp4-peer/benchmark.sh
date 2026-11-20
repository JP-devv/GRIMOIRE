#!/usr/bin/env bash
# Paired frozen TP4 benchmark. This is not a PP4/TP4 parity gate.
set -eo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
bin=$(realpath -e -- "${1:?isolated JIT binary}")
prompt_file=$(realpath -e -- "${2:?frozen prompt IDs}")
out=$(realpath -e -- "${3:?fresh output directory}")
[[ -x $bin && -d $out && -z $(ls -A -- "$out") ]]
job=${out##*/}
stage=inputs
socket_dir=
exec > "$out/run.log" 2>&1
trap 'rc=$?; if [[ -n $socket_dir ]]; then rm -f -- "$socket_dir/p"; rmdir -- "$socket_dir"; fi; printf "stage=%s exit=%s\n" "$stage" "$rc" > "$out/receipt.tmp"; mv "$out/receipt.tmp" "$out/receipt"; if [[ -n ${TMUX:-} ]]; then tmux wait-for -S "$job-done"; fi' EXIT
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null
set -u
model=${QFN_MODEL:?set QFN_MODEL to the dense text snapshot}
ple=${QFN_PLE_SOURCE:?set QFN_PLE_SOURCE to the original snapshot with PLE}
[[ -x $root/bin/qfn_parity_receipt ]]
printf '%s  %s\n' ac11a5713474144537b456939087893df81f650f3433531d7bba2952b1e1c6a6 "$prompt_file" | sha256sum -c -
sha256sum "$bin" "$prompt_file" "$model/model.safetensors.index.json" "$ple/model.safetensors.index.json" > "$out/inputs.sha256"
if [[ -n ${QFN_SERVING_BIN:-} ]]; then sha256sum "$QFN_SERVING_BIN" > "$out/serving.sha256"; fi
prompt=$(< "$prompt_file")
for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
unset ZE_AFFINITY_MASK UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS
affinity=(ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0)
export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
export GRIMOIRE_QFN_PLE_SOURCE=$ple GRIMOIRE_PLE_RAM=0 GRIMOIRE_MTP=0 GRIMOIRE_PREFIX_CACHE=0
export GRIMOIRE_PREFILL_CHUNK=512 GRIMOIRE_QFN_PREFILL_GEMV=1 GRIMOIRE_QFN_MOE_XMX=0
export GRIMOIRE_PROMPT_IDS=$prompt GRIMOIRE_PRINT_IDS=1 GRIMOIRE_P2P_CHUNK_MB=8
stage=lock
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -w "${QFN_LOCK_WAIT:-90}" 9
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
run() {
    local name=$1 peer=$2 rc=0 status rank
    shift 2
    local -a pids=()
    stage=$name
    mkdir "$out/$name"
    socket_dir=$(mktemp -d /tmp/qfn-tp4-ab.XXXXXXXX)
    printf '%s start %s\n' "$(date -Is)" "$name"
    for rank in 3 2 1 0; do
        env "${affinity[@]}" "$@" "GRIMOIRE_P2P=$peer" "GRIMOIRE_TP_RANK=$rank" "GRIMOIRE_TP_SOCKET=$socket_dir/p" \
            timeout --signal=TERM "${QFN_GPU_TIMEOUT:-240s}" "$bin" -m "$model" \
            --proj bf16 --ctx 32768 -p frozen -n 32 \
            > "$out/$name/matched-r$rank.out" 2> "$out/$name/matched-r$rank.log" &
        pids[rank]=$!
    done
    for rank in 0 1 2 3; do
        status=0
        wait "${pids[rank]}" || status=$?
        printf '%s rank=%d exit=%d\n' "$name" "$rank" "$status"
        if (( rc == 0 && status != 0 )); then rc=$status; fi
    done
    rm -f -- "$socket_dir/p"
    rmdir -- "$socket_dir"
    socket_dir=
    (( rc == 0 )) || return "$rc"
    for rank in 0 1 2 3; do
        local stem="$out/$name/matched-r$rank"
        grep -Fxq "prompt ids (2706): $prompt" "$stem.log"
        grep '^greedy ids (32): ' "$stem.log" > "$stem.ids"
        if (( peer )); then
            grep -Fq 'TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$stem.out"
        fi
    done
}
run host-gate 0 GRIMOIRE_QFN_LOGITS_AT_POS=2706 GRIMOIRE_BUDGET_DEBUG=1
run peer-gate 1 GRIMOIRE_QFN_LOGITS_AT_POS=2706 GRIMOIRE_BUDGET_DEBUG=1
stage=parity
"$root/bin/qfn_parity_receipt" --tp4 "$out" 32 | tee "$out/parity.log"
printf 'pair\ttransport\tprefill_ms\tdecode_ms\tprefill_tok_s\tdecode_tok_s\n' > "$out/timing.tsv"
for arm in 1-host 1-peer 2-peer 2-host; do
    kind=${arm#*-}
    peer=0; [[ $kind != peer ]] || peer=1
    run "$arm" "$peer" GRIMOIRE_GENERATION_PHASES=1
    for rank in 0 1 2 3; do
        cmp "$out/host-gate/matched-r$rank.ids" "$out/$arm/matched-r$rank.ids"
    done
    # Use the slowest rank's span per phase, not an average of GPU ranks.
    awk -v pair="${arm%%-*}" -v kind="$kind" '
        /^generation phases: / {
            if (NF != 6 || $3 != "prefill_tokens=2706" || $5 != "decode_tokens=31" || ++seen[FILENAME] != 1) bad=1
            if ($4 !~ /^prefill_ms=[0-9]+\.[0-9]+$/ || $6 !~ /^decode_ms=[0-9]+\.[0-9]+$/) bad=1
            p=substr($4,12)+0; d=substr($6,11)+0
            if (p<=0 || d<=0) bad=1
            if (p>prefill) prefill=p
            if (d>decode) decode=d
            ++ranks
        }
        END {
            if (bad || ranks!=4) exit 1
            printf "%s\t%s\t%.3f\t%.3f\t%.2f\t%.2f\n", pair, kind, prefill, decode, 2706000/prefill, 31000/decode
        }
    ' "$out/$arm"/matched-r*.log | tee -a "$out/timing.tsv"
done
stage=verify
sha256sum -c "$out/inputs.sha256"
if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
