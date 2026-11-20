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
        grep -Fxq "prompt ids (2706): $prompt" "$stem.log" || return
        if [[ $mode != verifier-diagnostic ]]; then
            grep '^greedy ids (32): ' "$stem.log" > "$stem.ids" || return
        fi
        if (( peer )); then
            grep -Fq 'TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$stem.out" || return
        fi
    done
}
gate=(GRIMOIRE_QFN_LOGITS_AT_POS=2706 GRIMOIRE_BUDGET_DEBUG=1)
mode=${QFN_BENCH_MODE:-transport}
case $mode in
    chunk-sweep)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        printf 'chunk\trank\tcalls\tsub_chunks\treductions\n' > "$out/reductions.tsv"
        arms=()
        for chunk in 512 1024 2048; do
            size="GRIMOIRE_PREFILL_CHUNK=$chunk"
            if ! run "$chunk-gate" 1 "$size" GRIMOIRE_Q4_TIMING=1 GRIMOIRE_QFN_MOE_TRACE=1; then
                printf 'chunk=%d failed gate execution, skipping timing\n' "$chunk"
                continue
            fi
            stage="$chunk-parity"
            valid=1
            for rank in 0 1 2 3; do
                if ! cmp "$frozen/matched-r$rank.ids" "$out/$chunk-gate/matched-r$rank.ids"; then
                    printf 'chunk=%d rank=%d differs from frozen TP4 IDs\n' "$chunk" "$rank"
                    valid=0
                fi
            done
            # The pinned binary has no reduction counter for multi-row calls.
            # Count from emitted call/sub-chunk shapes: 48 mixer and 1 PLE
            # reduction per sub-chunk; 48 MoE reductions per full call.
            if ! awk -v chunk="$chunk" '
                function finish() { if (tokens != 2706) bad=1 }
                FNR==1 {
                    if (NR>1) finish()
                    rank=files++; tokens=0
                }
                /^  qwen4_exp prefill M=[0-9]+ \(sub-chunks of [0-9]+\):/ {
                    m=substr($3,3)+0; s=$6+0
                    remaining=2706-tokens
                    if (m != (remaining<chunk ? remaining : chunk) || s<1 || s>m) { bad=1; next }
                    tokens+=m; calls[rank]++; subs[rank]+=int((m+s-1)/s)
                }
                END {
                    finish()
                    if (bad || files!=4) exit 1
                    for (i=0; i<4; ++i)
                        printf "%d\t%d\t%d\t%d\t%d\n", chunk,i,calls[i],subs[i],48*calls[i]+49*subs[i]
                }
            ' "$out/$chunk-gate"/matched-r*.log >> "$out/reductions.tsv"; then
                printf 'chunk=%d did not execute the requested shapes, skipping timing\n' "$chunk"
                valid=0
            fi
            if (( valid )); then
                printf 'chunk=%d matched all 32 frozen IDs on all four ranks\n' "$chunk"
                arms+=("1-$chunk")
            fi
        done
        (( ${#arms[@]} ))
        ;;
    transport)
        run host-gate 0 "${gate[@]}"
        run peer-gate 1 "${gate[@]}"
        stage=parity
        "$root/bin/qfn_parity_receipt" --tp4 "$out" 32 | tee "$out/parity.log"
        arms=(1-host 1-peer 2-peer 2-host)
        ;;
    grouped-offset)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_PREFILL_CHUNK=2048
        export GRIMOIRE_QFN_DEVICE_ROUTES=1 GRIMOIRE_QFN_TP_SYNC=fanout
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        candidate=GRIMOIRE_QFN_GROUPED_OFFSET_SCHED=1
        run baseline-gate 1 GRIMOIRE_QFN_MOE_TRACE=1
        run offset-gate 1 GRIMOIRE_QFN_MOE_TRACE=1 "$candidate"
        stage=grouped-offset-parity
        for kind in baseline offset; do
            for rank in 0 1 2 3; do
                stem="$out/$kind-gate/matched-r$rank"
                grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$stem.out"
                diff -u "$frozen/matched-r$rank.ids" "$stem.ids"
                awk '/^  QFN MoE XMX mode=2 .*device_routes=1 wide_n=0$/ {print $5,$6,$7}' "$stem.log" |
                    diff -u <(printf 'prefill_rows=2048 dispatches=48 gemm_launches=96\nprefill_rows=658 dispatches=48 gemm_launches=96\n') -
                awk -v expected="$([[ $kind == offset ]] && echo 2 || echo 0)" '
                    /^QFN grouped offset scheduler:/ {
                        if ($5 != "projection=gate_up" || $6 != "rows=" (++lines == 1 ? 2048 : 658) ||
                            $7 != "calls=48" || $8 !~ /^workers=[1-9][0-9]*$/ || (lines>1 && $8!=workers)) bad=1
                        workers=$8
                    }
                    END {if (bad || lines != expected) exit 1}
                ' "$stem.log"
            done
        done
        arms=(1-baseline 1-offset 2-offset 2-baseline)
        ;;
    grouped-prefetch)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_PREFILL_CHUNK=2048
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        candidate=GRIMOIRE_QFN_GROUPED_K_PREFETCH=1
        run baseline-gate 1 GRIMOIRE_QFN_TP_SYNC=fanout GRIMOIRE_QFN_MOE_TRACE=1
        run prefetch-gate 1 GRIMOIRE_QFN_TP_SYNC=fanout GRIMOIRE_QFN_MOE_TRACE=1 "$candidate"
        stage=grouped-prefetch-parity
        for kind in baseline prefetch; do
            for rank in 0 1 2 3; do
                stem="$out/$kind-gate/matched-r$rank"
                grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$stem.out"
                diff -u "$frozen/matched-r$rank.ids" "$stem.ids"
                awk '/^  QFN MoE XMX mode=2 .*device_routes=0 wide_n=0$/ {print $5,$6,$7}' "$stem.log" |
                    diff -u <(printf 'prefill_rows=2048 dispatches=48 gemm_launches=96\nprefill_rows=658 dispatches=48 gemm_launches=96\n') -
                if [[ $kind == prefetch ]]; then
                    awk '/^QFN grouped K prefetch: / {print $5,$6,$7}' "$stem.log" |
                        diff -u <(printf 'projection=gate_up rows=2048 calls=48\nprojection=gate_up rows=658 calls=48\n') -
                fi
            done
        done
        arms=(1-baseline 1-prefetch 2-prefetch 2-baseline)
        ;;
    grouped-projections)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_PREFILL_CHUNK=2048
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        run projection-profile 1 GRIMOIRE_QFN_TP_SYNC=fanout GRIMOIRE_QFN_GROUPED_PROFILE=1
        stage=projection-profile-parity
        for rank in 0 1 2 3; do
            stem="$out/projection-profile/matched-r$rank"
            diff -u "$frozen/matched-r$rank.ids" "$stem.ids"
            awk -v rank="$rank" '
                /^QFN grouped projection:/ {
                    ++lines
                    if ($4 != "rank=" rank || $5 != "rows=" (lines==1 ? 2048 : 658) ||
                        $6 != "gu_calls=48" || $8 != "down_calls=48" ||
                        $7 !~ /^gu_ms=[0-9]+\.[0-9]+$/ || $9 !~ /^down_ms=[0-9]+\.[0-9]+$/ ||
                        substr($7,7)+0 <= 0 || substr($9,9)+0 <= 0) bad=1
                    print
                }
                END {if (bad || lines!=2) exit 1}
            ' "$stem.log"
        done | tee "$out/projection-profile.log"
        stage=verify
        sha256sum -c "$out/inputs.sha256"
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        exit 0
        ;;
    ipc-events)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_PREFILL_CHUNK=2048
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        sync_parity() {
            local arm=$1 kind=$2 rank stem mode_banner
            stage="$arm-parity"
            mode_banner='socket fan-out'
            if [[ $kind == ipc-events ]]; then mode_banner='Level Zero IPC device events'; fi
            for rank in 0 1 2 3; do
                stem="$out/$arm/matched-r$rank"
                grep -Fxq "  QFN TP4 synchronization: $mode_banner" "$stem.out" || return
                diff -u "$frozen/matched-r$rank.ids" "$stem.ids" || return
            done
        }
        for kind in fanout ipc-events; do
            run "$kind-gate" 1 "GRIMOIRE_QFN_TP_SYNC=$kind" GRIMOIRE_QFN_MOE_TRACE=1 GRIMOIRE_Q4_TIMING=1
            sync_parity "$kind-gate" "$kind"
            for rank in 0 1 2 3; do
                awk '/^  qwen4_exp prefill M=[0-9]+ \(sub-chunks of / { print $3, $6 }' \
                    "$out/$kind-gate/matched-r$rank.log" |
                    diff -u <(printf 'M=2048 2048):\nM=658 658):\n') -
            done
        done
        arms=(1-fanout 1-ipc-events 2-ipc-events 2-fanout)
        ;;
    mtp3-diagnostic)
        draft=$(realpath -e -- "${QFN_DRAFT_SOURCE:?set QFN_DRAFT_SOURCE}")
        frozen=${QFN_GROUPED_CONTROL:?set frozen grouped control directory}
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_DECODE_GRAPH=0
        sha256sum "$draft"/{config.json,model.safetensors.index.json,mtp-dense.safetensors,mtp-routed-experts-int4.safetensors} >> "$out/inputs.sha256"
        mtp3=(GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=mtp3 "GRIMOIRE_QFN_MTP_SOURCE=$draft" GRIMOIRE_SPEC_STATS=1)
        # The CLI prints stats before flushing the decoder's text. On a
        # rank's stdout the stats can follow the last text without a newline.
        mtp3_stats='  spec: MTP depth 3 .* over [1-9][0-9]* steps$'
        mtp3_parity() {
            local arm=$1 kind=$2 rank stem
            stage="$arm-parity"
            for rank in 0 1 2 3; do
                stem="$out/$arm/matched-r$rank"
                diff -u "$frozen/matched-r$rank.ids" "$stem.ids" || return
                if [[ $kind == mtp-on ]]; then
                    grep -Eq "$mtp3_stats" "$stem.out" || return
                fi
            done
        }
        run mtp-off-gate 1 GRIMOIRE_BUDGET_DEBUG=1
        mtp3_parity mtp-off-gate mtp-off
        run mtp-on-gate 1 GRIMOIRE_BUDGET_DEBUG=1 "${mtp3[@]}"
        mtp3_parity mtp-on-gate mtp-on
        arms=(1-mtp-off 1-mtp-on 2-mtp-on 2-mtp-off)
        ;;
    mtp-paired)
        draft=$(realpath -e -- "${QFN_DRAFT_SOURCE:?set QFN_DRAFT_SOURCE}")
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        run mtp-off 1 GRIMOIRE_GENERATION_PHASES=1
        stage=mtp-off-parity
        for rank in 0 1 2 3; do
            cmp "${QFN_GROUPED_CONTROL:?set frozen grouped control directory}/matched-r$rank.ids" \
                "$out/mtp-off/matched-r$rank.ids"
        done
        awk '/^generation phases: /' "$out/mtp-off"/matched-r*.log > "$out/mtp-off-phases.log"
        run mtp-on 1 GRIMOIRE_GENERATION_PHASES=1 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 \
            "GRIMOIRE_QFN_MTP_SOURCE=$draft" GRIMOIRE_SPEC_STATS=1
        stage=mtp-verify
        for rank in 0 1 2 3; do
            cmp "$out/mtp-off/matched-r$rank.ids" "$out/mtp-on/matched-r$rank.ids"
            grep -E '^  spec: MTP depth 3 .* over [1-9][0-9]* steps$' "$out/mtp-on/matched-r$rank.out"
        done | tee "$out/mtp-spec.log"
        awk '/^generation phases: /' "$out/mtp-on"/matched-r*.log > "$out/mtp-on-phases.log"
        stage=verify
        sha256sum -c "$out/inputs.sha256"
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        exit 0
        ;;
    mtp-red)
        # A TP4 draft must run a real verifier, not just load or silently fall
        # back to target-only. The baseline refuses before reaching this gate.
        draft=$(realpath -e -- "${QFN_DRAFT_SOURCE:?set QFN_DRAFT_SOURCE}")
        run mtp-gate 0 "${gate[@]}" GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 \
            "GRIMOIRE_QFN_MTP_SOURCE=$draft" GRIMOIRE_SPEC_STATS=1
        for rank in 0 1 2 3; do
            grep -E 'spec: MTP depth 3 .* over [1-9][0-9]* steps' "$out/mtp-gate/matched-r$rank.out"
        done
        exit 0
        ;;
    mtp-preflight)
        rc=0
        run mtp-load 0 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 || rc=$?
        stage=mtp-unsupported
        (( rc != 0 )) || exit 2
        for rank in 0 1 2 3; do
            grep -F 'GRIMOIRE_QFN_TP4 needs four TP ranks, dense safetensors, --proj bf16, and speculation off' \
                "$out/mtp-load/matched-r$rank.log"
        done
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        exit 0
        ;;
    hc)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        case ${QFN_HC_TEST:-fused} in
            fused) hc_case=fused; hc_flag=GRIMOIRE_QFN_HC_FUSED_PREFILL ;;
            tiled)
                hc_case=tiled; hc_flag=GRIMOIRE_QFN_HC_DOWN_TILED
                stage=hc-bitwise
                env "${affinity[@]}" timeout --signal=TERM 90s "${bin%/*}/test_hc_down_tiles" | tee "$out/hc-bitwise.log"
                ;;
            *) exit 2 ;;
        esac
        run unfused-gate 1 "${gate[@]}"
        run fused-gate 1 "${gate[@]}" "$hc_flag=1"
        for rank in 0 1 2 3; do
            [[ $(grep -c "^QFN HC prefill $hc_case .*calls=[1-9]" "$out/fused-gate/matched-r$rank.log") == 6 ]]
        done
        stage=parity
        "$root/bin/qfn_parity_receipt" --hc "$out" 32 | tee "$out/parity.log"
        if [[ $hc_case == tiled ]]; then
            for rank in 0 1 2 3; do
                cmp "${QFN_GROUPED_CONTROL:?set frozen grouped control directory}/matched-r$rank.ids" \
                    "$out/unfused-gate/matched-r$rank.ids"
            done
        fi
        arms=(1-unfused 1-fused 2-fused 2-unfused)
        ;;
    verifier-diagnostic)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        rc=0
        run tp4-verifier 1 GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=1 || rc=$?
        stage=tp4-verifier-same-batch
        awk '/^QFN verify / || /^generation failed: / || /^load: /' \
            "$out/tp4-verifier"/matched-r*.log | tee "$out/verifier.log"
        sha256sum -c "$out/inputs.sha256"
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        (( rc == 0 || rc == 2 )) || exit "$rc"
        for rank in 0 1 2 3; do
            grep -Fxq "QFN verify same-batch result rank=$rank prefix_tokens=2708 restored_next=864 direct_next=864 commit_exact=1 repeat_checkpoint_exact=1 correction_exact=1 pass=1" \
                "$out/tp4-verifier/matched-r$rank.log"
        done
        stage=tp4-verifier-sequential-compare
        (( rc == 0 )) || exit "$rc"
        for rank in 0 1 2 3; do
            grep -Fxq "QFN verify sequential result rank=$rank prefix_tokens=2708 restored_next=864 target_next=864 tokens_match=1 exact_state_and_logits=1 pass=1" \
                "$out/tp4-verifier/matched-r$rank.log"
        done
        stage=verify
        exit 0
        ;;
    decode-fanout)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        run grouped-gate 1 GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1
        run fanout-gate 1 GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1 GRIMOIRE_QFN_TP_DECODE_FANOUT=1
        stage=decode-fanout-parity
        for rank in 0 1 2 3; do
            frozen="${QFN_GROUPED_CONTROL:?set frozen grouped control directory}/matched-r$rank.ids"
            cmp "$frozen" "$out/grouped-gate/matched-r$rank.ids"
            cmp "$frozen" "$out/fanout-gate/matched-r$rank.ids"
            grep -Fxq '  QFN TP4 decode: fan-out control packets' "$out/fanout-gate/matched-r$rank.out"
            grep -Fxq "QFN TP decode payload rank=$rank kind=moe calls=1488 bytes=15237120 min_bytes=10240 max_bytes=10240" \
                "$out/fanout-gate/matched-r$rank.log"
        done
        printf 'PASS: all 32 frozen TP4 IDs on all ranks in both decode arms\n' | tee "$out/parity.log"
        arms=(1-grouped 1-fanout 2-fanout 2-grouped)
        ;;
    decode-profile)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        run decode-profile 1 GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1
        stage=decode-profile-verify
        for rank in 0 1 2 3; do
            stem="$out/decode-profile/matched-r$rank"
            cmp "${QFN_GROUPED_CONTROL:?set frozen grouped control directory}/matched-r$rank.ids" "$stem.ids"
            grep -E "^QFN TP decode (projections rank=$rank tokens=31 |(reduce|payload) rank=$rank kind=)" "$stem.log"
            grep -q "^QFN TP decode projections rank=$rank tokens=31 " "$stem.log"
            # 31 steps: one PLE, 36 DeltaNet, 12 attention, and 48 MoE
            # reductions each; every logical payload is 2560 FP32 values.
            for entry in ple:1 deltanet:36 attention:12 moe:48; do
                calls=$((31 * ${entry#*:}))
                grep -Fxq "QFN TP decode payload rank=$rank kind=${entry%:*} calls=$calls bytes=$((calls*10240)) min_bytes=10240 max_bytes=10240" "$stem.log"
            done
        done | tee "$out/decode-profile.log"
        stage=verify
        sha256sum -c "$out/inputs.sha256"
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        exit 0
        ;;
    profile)
        export GRIMOIRE_QFN_PREFILL_GEMV=0
        detail=()
        case ${QFN_PROFILE_DETAIL:-0} in
            0) ;;
            1) detail=(GRIMOIRE_Q4_TIMING=1) ;;
            2) detail=(GRIMOIRE_HC_TIMING=1) ;;
            3) detail=(GRIMOIRE_Q4_TIMING=1 GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1) ;;
            *) exit 2 ;;
        esac
        run grouped-profile 1 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_TP_PREFILL_REDUCE_TIMING=1 GRIMOIRE_GENERATION_PHASES=1 "${detail[@]}"
        stage=profile
        awk '
            BEGIN { OFS="\t"; print "rank", "calls", "span_ms", "communication_ms", "routed_plan_ms", "routed_device_ms", "rest_ms" }
            /^QFN TP prefill profile / {
                split($5, a, "="); r=a[2]; ++chunks[r]
                for (i=4; i<=10; ++i) { split($(i+4), a, "="); v[i]=a[2]+0 }
                if (v[5]!=48 || v[4]!=97 || v[10]<0) bad=1
                for (i=4; i<=10; ++i) sums[r,i]+=v[i]
            }
            END {
                for (r=0; r<4; ++r) {
                    if (chunks[r]!=6) bad=1
                    print r, sums[r,4], sums[r,6], sums[r,7], sums[r,8], sums[r,9], sums[r,10]
                }
                if (bad) exit 1
            }
        ' "$out/grouped-profile"/matched-r*.log | tee "$out/profile.tsv"
        stage=verify
        sha256sum -c "$out/inputs.sha256"
        if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
        exit 0
        ;;
    grouped-width|grouped-routes)
        export GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2
        pair_name=wide; candidate=GRIMOIRE_QFN_GROUPED_WIDE_N=1; tag='wide_n=1$'
        if [[ $mode == grouped-routes ]]; then
            pair_name=device; candidate=GRIMOIRE_QFN_DEVICE_ROUTES=1; tag='device_routes=1 '
            stage=device-route-fixture
            env "${affinity[@]}" timeout --signal=TERM 90s "${bin%/*}/test_qfn_device_routes" | tee "$out/device-routes.log"
        fi
        run grouped-gate 1 "${gate[@]}" GRIMOIRE_QFN_MOE_TRACE=1
        run "$pair_name-gate" 1 "${gate[@]}" GRIMOIRE_QFN_MOE_TRACE=1 "$candidate"
        stage=parity
        for rank in 0 1 2 3; do
            [[ $(grep -c "^  QFN MoE XMX mode=2 .*$tag" "$out/$pair_name-gate/matched-r$rank.log") == 6 ]]
            cmp "${QFN_GROUPED_CONTROL:?set frozen grouped control directory}/matched-r$rank.ids" \
                "$out/grouped-gate/matched-r$rank.ids"
        done
        "$root/bin/qfn_parity_receipt" "--$mode" "$out" 32 | tee "$out/parity.log"
        arms=(1-grouped "1-$pair_name" "2-$pair_name" 2-grouped)
        ;;
    grouped)
        export GRIMOIRE_QFN_PREFILL_GEMV=0
        run dense-gate 1 "${gate[@]}"
        run grouped-gate 1 "${gate[@]}" GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_MOE_TRACE=1
        for rank in 0 1 2 3; do
            [[ $(grep -c '^  QFN MoE XMX mode=2 .*dispatches=[1-9]' "$out/grouped-gate/matched-r$rank.log") == 6 ]]
        done
        stage=parity
        "$root/bin/qfn_parity_receipt" --grouped "$out" 32 | tee "$out/parity.log"
        arms=(1-dense 1-grouped 2-grouped 2-dense)
        ;;
    *) printf 'unknown QFN_BENCH_MODE: %s\n' "$mode" >&2; exit 2 ;;
esac
if [[ $mode != chunk-sweep && ${QFN_BENCH_PAIRS:-2} == 1 ]]; then arms=("${arms[@]:0:2}"); fi
printf 'pair\tarm\tprefill_ms\tdecode_ms\tprefill_tok_s\tdecode_tok_s\n' > "$out/timing.tsv"
for arm in "${arms[@]}"; do
    kind=${arm#*-}
    peer=1; [[ $kind != host ]] || peer=0
    xmx=0; hc=0
    if [[ $kind == grouped || $kind == wide || $kind == device || $mode == hc ]]; then xmx=2; fi
    extra=()
    if [[ $kind == fused ]]; then hc=1; extra=("$hc_flag=$hc"); fi
    if [[ $kind == wide || $kind == device ]]; then extra=("$candidate"); fi
    if [[ $kind == fanout ]]; then xmx=2; extra=(GRIMOIRE_QFN_TP_DECODE_FANOUT=1); fi
    if [[ $mode == chunk-sweep ]]; then xmx=2; extra=("GRIMOIRE_PREFILL_CHUNK=$kind"); fi
    if [[ $mode == ipc-events ]]; then
        xmx=2; extra=("GRIMOIRE_QFN_TP_SYNC=$kind")
    fi
    if [[ $mode == grouped-prefetch || $mode == grouped-offset ]]; then
        xmx=2; extra=(GRIMOIRE_QFN_TP_SYNC=fanout)
        if [[ $kind == prefetch || $kind == offset ]]; then extra+=("$candidate"); fi
    fi
    if [[ $mode == mtp3-diagnostic ]]; then
        xmx=2
        if [[ $kind == mtp-on ]]; then extra=("${mtp3[@]}"); fi
    fi
    run "$arm" "$peer" GRIMOIRE_GENERATION_PHASES=1 GRIMOIRE_QFN_MOE_XMX="$xmx" "${extra[@]}"
    if [[ $mode == ipc-events ]]; then
        sync_parity "$arm" "$kind"
    elif [[ $mode == mtp3-diagnostic ]]; then
        mtp3_parity "$arm" "$kind"
        if [[ $kind == mtp-on ]]; then
            grep -EHo "$mtp3_stats" "$out/$arm"/matched-r*.out | tee -a "$out/acceptance.log"
        fi
    else
        for rank in 0 1 2 3; do
            cmp "$out/$kind-gate/matched-r$rank.ids" "$out/$arm/matched-r$rank.ids"
        done
    fi
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
if [[ $mode == mtp3-diagnostic ]]; then
    # Fenced phase latencies are deliberately separate from throughput.
    run mtp-on-profile 1 "${mtp3[@]}" GRIMOIRE_SPEC_TIME=1
    mtp3_parity mtp-on-profile mtp-on
    stage=mtp-on-profile-latency
    for rank in 0 1 2 3; do
        grep -H '^  spec time per step (ms, fenced): ' "$out/mtp-on-profile/matched-r$rank.out"
    done | tee "$out/spec-latency.log"
fi
stage=verify
sha256sum -c "$out/inputs.sha256"
if [[ -e $out/serving.sha256 ]]; then sha256sum -c "$out/serving.sha256"; fi
