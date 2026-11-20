#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
out=$(realpath -- "${1:?fresh output directory}")
source_chunk=$(realpath -- "${2:?PP4 layer entry capture}")
frozen=$(realpath -- "${3:?frozen prompt receipt}")
cd "$script_dir/../.."
layer=${4:-1}
# Refuse to reuse receipts, logs, or GPU captures from a previous run.
[[ -d $out && -z $(ls -A -- "$out") ]]
stage=build
build_exit=not-run
binary_sha256=unavailable
job=${out##*/}
bin=$PWD/bin/grimoire-qfn-dn-router-jit-$job
exec > "$out/replay.log" 2>&1
trap 'rc=$?; printf "stage=%s exit=%s build_exit=%s binary_sha256=%s\n" "$stage" "$rc" "$build_exit" "$binary_sha256" > "$out/receipt.tmp"; mv "$out/receipt.tmp" "$out/receipt"; tmux wait-for -S "$job-done"' EXIT
printf '%s start build\n' "$(date -Is)"
if [[ -e $bin ]]; then printf 'refusing existing binary: %s\n' "$bin"; exit 1; fi
build_exit=0
timeout --kill-after=10s "${QFN_BUILD_TIMEOUT:-600s}" bash "$script_dir/build-dn-jit.sh" "$bin" "$out/obj" > "$out/build.log" 2>&1 || build_exit=$?
printf '%s\n' "$build_exit" > "$out/build.exit"
(( build_exit == 0 )) || exit "$build_exit"
sha256sum "$bin" > "$out/rebuilt-binary.sha256"
read -r binary_sha256 _ < "$out/rebuilt-binary.sha256"
sha256sum -c "$out/rebuilt-binary.sha256"
stage=checker
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null
set -u
model=${QFN_MODEL:?set QFN_MODEL to the dense text snapshot}
ple_source=${QFN_PLE_SOURCE:?set QFN_PLE_SOURCE to the original snapshot containing PLE}
# The inputs are taken from the owning stage's combined layer entry, before PLE.
case "$layer" in
    0) offset=20971548 ;; # layer-0 entry follows the embedding record
    1) offset=0 ;;        # layer-1 detail capture begins with its entry record
    *) exit 2 ;;
esac
read -ra fields <<< "$(od -An -v -tu4 -j "$offset" -N28 -w28 -- "$source_chunk")"
[[ ${fields[*]} == "826361425 1 $layer 0 0 512 10240" ]]
dd if="$source_chunk" of="$out/entry.f32" iflag=skip_bytes,count_bytes bs=1M skip=$((offset + 28)) count=20971520 status=none
[[ $(stat -c %s "$out/entry.f32") == 20971520 ]]
sha256sum "$out/entry.f32"
printf '%s  %s\n' ac11a5713474144537b456939087893df81f650f3433531d7bba2952b1e1c6a6 "$frozen" | sha256sum -c -
awk 'NR == 1 && NF == 2706 {for(i=1;i<=512;++i) printf "%s%s", $i, i==512? "\n" : " "; found=1} END {exit !found || NR != 1}' "$frozen" > "$out/prompt-512.ids"

stage=lock
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -w 90 9
render=(/dev/dri/renderD*)
[[ ${#render[@]} == 4 && -c ${render[0]} ]]
command -v fuser >/dev/null
sudo -n true
probe_rc=0
probe=$(sudo -n fuser "${render[@]}" 2>&1) || probe_rc=$?
if [[ $probe_rc != 1 || -n $probe ]]; then
    printf 'GPU occupancy check refused: rc=%d %s\n' "$probe_rc" "$probe"
    exit 1
fi

unset UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS GRIMOIRE_DEVICES ZE_AFFINITY_MASK GRIMOIRE_PLE_FILE
unset GRIMOIRE_PP_RANK GRIMOIRE_TP_RANK GRIMOIRE_PP_WORLD_SIZE GRIMOIRE_TP_WORLD_SIZE GRIMOIRE_QFN_TP4 GRIMOIRE_QFN_TP4_TRACE GRIMOIRE_TP_MODE
unset GRIMOIRE_DFLASH_MODEL GRIMOIRE_DFLASH2_MODEL GRIMOIRE_QWEN4EXP_SEQUENTIAL_PREFILL
unset GRIMOIRE_QFN_ACT_DIR GRIMOIRE_QFN_ACT_LAYER GRIMOIRE_QFN_MOE_ROW_INPUT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 GRIMOIRE_PRINT_IDS=1 GRIMOIRE_P2P=0
export GRIMOIRE_PLE_RAM=0 GRIMOIRE_QFN_PLE_SOURCE=$ple_source GRIMOIRE_MTP=0
export GRIMOIRE_QFN_PREFILL_GEMV=1 GRIMOIRE_QFN_MOE_XMX=0 GRIMOIRE_PREFILL_CHUNK=512 GRIMOIRE_PREFIX_CACHE=0
export GRIMOIRE_QFN_MOE_ROW_LAYER=$layer GRIMOIRE_QFN_MOE_ROW_POS=361 GRIMOIRE_QFN_MOE_ROW_ENTRY=$out/entry.f32
export GRIMOIRE_QFN_MOE_ROW_STOP=1
export GRIMOIRE_PROMPT_IDS=$(<"$out/prompt-512.ids")
run() {
    local layout=$1 role=$2 r sockets rank_rc rc=0
    shift 2
    local -a pids=()
    mkdir "$out/$layout"
    sockets=$(mktemp -d /tmp/qfn-dn.XXXXXXXX)
    printf '%s start %s\n' "$(date -Is)" "$stage"
    for r in 3 2 1 0; do
        env "$@" GRIMOIRE_QFN_MOE_ROW_DIR="$out/$layout" "GRIMOIRE_${role}_RANK=$r" "GRIMOIRE_${role}_SOCKET=$sockets/p" \
            timeout --signal=TERM 240s "$bin" -m "$model" --proj bf16 --ctx 32768 -p diagnostic -n 1 \
            > "$out/$layout/r$r.stdout" 2> "$out/$layout/r$r.log" &
        pids[r]=$!
        sleep 1
    done
    for r in 0 1 2 3; do
        rank_rc=0
        wait "${pids[r]}" || rank_rc=$?
        printf '%s rank=%d exit=%d\n' "$stage" "$r" "$rank_rc"
        if (( rc == 0 && rank_rc != 0 )); then rc=$rank_rc; fi
    done
    rm -f "$sockets"/p "$sockets"/p-{1,2,3}
    rmdir "$sockets"
    if (( rc )); then return "$rc"; fi
    for r in 0 1 2 3; do
        rg -q '^QFN router-stop complete rows=512$' "$out/$layout/r$r.log" || return
        if [[ $layout == tp4 || $r == 0 ]]; then
            cmp "$out/entry.f32" "$out/$layout/l$layer-p361-r$r-entry-span.bin" || return
        fi
    done
}
stage=PP4
run pp4 PP GRIMOIRE_PP_WORLD_SIZE=4 GRIMOIRE_PP_LAYERS=12,12,12,12
stage=TP4
run tp4 TP GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
flock -u 9
stage=checker
bin/qfn_compare_dn_prefix "$out" "$layer" | tee "$out/comparison.log"
rg '^first_differing_tap ' "$out/comparison.log"
sha256sum -c "$out/rebuilt-binary.sha256"
if [[ -n ${QFN_SERVING_BIN:-} ]]; then sha256sum "$QFN_SERVING_BIN"; fi
