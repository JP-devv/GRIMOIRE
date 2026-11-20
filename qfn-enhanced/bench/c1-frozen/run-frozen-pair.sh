#!/usr/bin/env bash
set -euo pipefail
Q=$(cd -- "$(dirname -- "$0")/../.." && pwd)
B=$Q/bench/c1-frozen
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
D=$(mktemp -d "$Q/../receipts/frozen-pair.XXXXXXXX")
printf "Receipts %s\n" "$D"
IMAGE=sha256:bcc91a8966e805a632f575b0467d4b8f926e1f6d7ec36cf4ecc16f3627537a05
: "${APPROVED_DEADLINE_EPOCH:?}" "${BENCH_PORT:?}" "${TMUX:?}"
[[ $APPROVED_DEADLINE_EPOCH =~ ^[1-9][0-9]{9}$ && $BENCH_PORT =~ ^[1-9][0-9]{3,4}$ ]]
(( BENCH_PORT <= 65535 && BENCH_PORT != 19622 ))
(cd "$B"; sha256sum -c inputs.sha256; sha256sum -c native.sha256; python3 checkpoint-lock.py)
[[ $(docker image inspect --format '{{.Id}}' b70-flash-next:tp2) == "$IMAGE" ]]
sha256sum -c "$SERVING"
python3 -B "$B/render-fixtures.py" "$D/fixtures"
fixture=${FIXTURE:-frozen}
case "$fixture" in
    frozen|full-chunk) output=32 ;;
    long-decode) output=256 ;;
    *) echo "Unknown fixture: $fixture" >&2; exit 2 ;;
esac
cp "$D/fixtures/$fixture.request.json" "$D/request.json"
if [[ ${PAIR_ORDER:-} == boundary ]]; then
    [[ $fixture == full-chunk ]]
    jq '.max_tokens = 2' "$D/request.json" > "$D/boundary-request.json"
    mv "$D/boundary-request.json" "$D/request.json"
fi
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
idle() {
    local node occupancy rc=0
    local -a render=(/dev/dri/renderD*) nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
    [[ ${#render[@]} == 4 ]] || return 1
    for node in "${nodes[@]}"; do [[ -c $node ]] || return 1; done
    occupancy=$(sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
    [[ $rc == 1 && -z $occupancy ]]
}
remaining() {
    local left=$((APPROVED_DEADLINE_EPOCH - $(date +%s) - 25))
    (( left >= $1 )) || { echo 'Not enough authorized time' >&2; return 1; }
    printf '%s\n' "$left"
}
idle
CURSOR=$(sudo -n journalctl -k -q -n 1 --show-cursor --no-pager | awk '/^-- cursor:/ {print $3}')
[[ -n $CURSOR ]]

sampling_pid=
finish() {
    local rc=$?
    trap - EXIT
    if [[ -n $sampling_pid ]]; then
        kill -TERM "$sampling_pid" 2>/dev/null || :
        wait "$sampling_pid" || :
    fi
    idle >> "$D/restore.log" 2>&1 || rc=1
    sha256sum -c "$SERVING" >> "$D/restore.log" 2>&1 || rc=1
    echo "$rc" > "$D/exit"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
for dev in /sys/bus/pci/devices/0000:{0a,0e,14,18}:00.0; do
    printf "%s\n" "$dev"
    cat "$dev"/hwmon/hwmon*/power1_cap "$dev"/tile0/gt0/freq0/{min_freq,max_freq,cur_freq}
done > "$D/cards-before.txt" 2>&1
nice -n 15 python3 -u "$B/sample-clocks.py" > "$D/gpu-clocks.tsv" 2> "$D/gpu-clocks.log" &
sampling_pid=$!
native() (
    idle
    unset LD_PRELOAD LD_LIBRARY_PATH
    for key in $(compgen -e GRIMOIRE_); do unset "$key"; done
    unset ZE_AFFINITY_MASK SYCL_DEVICE_FILTER SYCL_UR_TRACE SYCL_PI_TRACE
    set +u
    source /opt/intel/oneapi/setvars.sh --force
    set -u
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
    export GRIMOIRE_TP_WORLD_SIZE=4 GRIMOIRE_QFN_TP4=1 GRIMOIRE_TP_SHM=0 GRIMOIRE_SEQ_SLOTS=1
    export GRIMOIRE_P2P=1 GRIMOIRE_P2P_CHUNK_MB=1 GRIMOIRE_QFN_TP_SYNC=fanout
    export GRIMOIRE_QFN_PLE_SOURCE=/home/haxor/models/qwen38-w4a16 GRIMOIRE_PLE_RAM=0
    export GRIMOIRE_MTP=0 GRIMOIRE_PREFIX_CACHE=0 GRIMOIRE_DECODE_GRAPH=0
    export GRIMOIRE_PREFILL_CHUNK=2048 GRIMOIRE_QFN_PREFILL_GEMV=0 GRIMOIRE_QFN_MOE_XMX=2 GRIMOIRE_QFN_DEVICE_ROUTES=1
    export GRIMOIRE_QFN_GROUPED_K_PREFETCH=0 GRIMOIRE_QFN_GROUPED_LIVE_TILES=1
    export GRIMOIRE_QFN_XE2_GATE_UP=fp32 GRIMOIRE_QFN_XE2_DOWN=1
    export GRIMOIRE_QFN_HC_DOWN_TILED=1 GRIMOIRE_QFN_HC_FUSED_PREFILL=1
    NATIVE_BIN=/tmp/qfn-catchup.pf1/verifier/qfn-verifier
    case "${NATIVE_MODE:-baseline}" in
        baseline)
            if [[ $fixture == long-decode ]]; then
                build="$Q/../receipts/long-decode-build"
                [[ $(< "$build/build-long.exit") == 0 ]]
                (cd "$build"; sha256sum -c long.sha256)
                NATIVE_BIN=$build/qfn-combined
                export GRIMOIRE_BENCH_IGNORE_EOS=1
            fi
            ;;
        replay)
            [[ $fixture == frozen && ${PAIR_ORDER:-} == native-only ]]
            build="$Q/../receipts/row-replay-build"
            [[ $(< "$build/build-replay.exit") == 0 ]]
            (cd "$build"; sha256sum -c replay.sha256)
            NATIVE_BIN=$build/qfn-combined
            export GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=replay
            ;;
        hc-off|hc-on)
            [[ $fixture == frozen && ${PAIR_ORDER:-} == native-only ]]
            build="$Q/../receipts/hc-fused-build"
            [[ $(< "$build/build-hc.exit") == 0 ]]
            (cd "$build"; sha256sum -c hc.sha256)
            NATIVE_BIN=$build/qfn-combined
            export GRIMOIRE_QFN_HC_FUSED_BF16=0
            if [[ $NATIVE_MODE == hc-on ]]; then export GRIMOIRE_QFN_HC_FUSED_BF16=1; fi
            ;;
        rsag-off|rsag-on)
            [[ $fixture == frozen && ${PAIR_ORDER:-} == native-only ]]
            build="$Q/../receipts/prefill-rsag-build"
            [[ $(< "$build/build-rsag.exit") == 0 ]]
            (cd "$build"; sha256sum -c rsag.sha256)
            NATIVE_BIN=$build/qfn-combined
            export GRIMOIRE_QFN_PREFILL_RSAG=0
            if [[ $NATIVE_MODE == rsag-on ]]; then export GRIMOIRE_QFN_PREFILL_RSAG=1; fi
            # With packed N64 and DecodeM8 off, this prints only aggregate
            # prefill dispatch counters. It enables no event or fence timing.
            export GRIMOIRE_QFN_MOE_TRACE=1
            ;;
        integrated-mtp3|integrated-profile)
            [[ $fixture == frozen && ${PAIR_ORDER:-} == native-only ]]
            build="$Q/../receipts/integrated-build"
            [[ $(< "$build/build-integrated.exit") == 0 ]]
            (cd "$build"; sha256sum -c integrated.sha256)
            NATIVE_BIN=$build/qfn-combined
            export GRIMOIRE_QFN_PREFILL_RSAG=1 GRIMOIRE_QFN_VERIFY_SPLIT_KV=1 GRIMOIRE_QFN_VERIFY_REQUIRE_SPLIT=1
            export GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=mtp3 GRIMOIRE_SPEC_STATS=1
            export GRIMOIRE_QFN_MTP_SOURCE=/home/haxor/models/qwen38-w4a16-mtp-snapshot
            export GRIMOIRE_QFN_MOE_TRACE=1
            if [[ $NATIVE_MODE == integrated-profile ]]; then
                export GRIMOIRE_Q4_TIMING=1 GRIMOIRE_HC_TIMING=1 GRIMOIRE_QFN_DN_PROFILE=1
                export GRIMOIRE_QFN_TP_PREFILL_REDUCE_TIMING=1 GRIMOIRE_QFN_VERIFY_PROFILE=1 GRIMOIRE_SPEC_TIME=1
            fi
            ;;
        *) echo "Invalid NATIVE_MODE" >&2; exit 2 ;;
    esac
    printf '%s\n' "${NATIVE_MODE:-baseline}" > "$D/native-mode"
    export GRIMOIRE_PROMPT_IDS="$(< "$D/fixtures/$fixture.prompt.ids")" GRIMOIRE_PRINT_IDS=1 GRIMOIRE_GENERATION_PHASES=1
    compgen -e GRIMOIRE_ | sort | while IFS= read -r key; do
        printf '%s=%s\n' "$key" "${!key}"
    done > "$D/native.env"
    CAP=$(remaining 90); (( CAP <= 165 )) || CAP=165
    SOCK=$(mktemp -d /tmp/qfn-c1-sock.XXXXXXXX)
    pids=(); trap 'for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done; wait || :; rm -f -- "$SOCK/p"; rmdir -- "$SOCK"' EXIT
    trap 'exit 143' TERM INT
    mkdir "$D/native"
    for rank in 3 2 1 0; do
        GRIMOIRE_TP_RANK=$rank GRIMOIRE_TP_SOCKET=$SOCK/p \
            timeout --signal=TERM --kill-after=4s "$CAP" "$NATIVE_BIN" \
            -m /home/haxor/models/qwen38-w4a16-snapshot --proj bf16 --ctx 32768 -p "$fixture" -n "$output" \
            > "$D/native/r$rank.out" 2> "$D/native/r$rank.log" &
        pids[rank]=$!
    done
    for rank in 0 1 2 3; do wait "${pids[rank]}"; unset 'pids[rank]'; done
    for f in "$D/native"/r?.out; do
        grep -Fxq '  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$f"
        grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$f"
    done
    if [[ ${NATIVE_MODE:-} == replay ]]; then
        for rank in 0 1 2 3; do
            awk -v rank="$rank" '
                $1=="QFN" && $2=="row" && $3=="replay" {
                    expected_rows = count==0 || count==3 ? 1 : 4
                    if ($4!="rank=" rank || $5!="sample=" count+0 ||
                        $6!="rows=" expected_rows || $8!="target_ids_match=1") exit 2
                    print
                    ++count
                }
                END {if (count!=4) exit 1}
            ' "$D/native/r$rank.log"
        done > "$D/row-replay.txt"
    elif [[ $fixture == frozen ]]; then
        if [[ ${NATIVE_MODE:-} == hc-on ]]; then
            for rank in 0 1 2 3; do
                grep -Eq "^QFN HC fused BF16 rank=$rank rows=2048 calls=[1-9][0-9]*$" "$D/native/r$rank.log"
            done
        fi
        if [[ ${NATIVE_MODE:-} == rsag-* || ${NATIVE_MODE:-} == integrated-* ]]; then
            for rank in 0 1 2 3; do
                grep -Eq "^QFN Xe2 gate/up: rank=$rank rows=2048 calls=48 output=fp32$" "$D/native/r$rank.log"
                grep -Eq "^QFN Xe2 down: rank=$rank rows=2048 calls=48 output=fp32$" "$D/native/r$rank.log"
                if [[ $NATIVE_MODE != rsag-off ]]; then
                    grep -Fxq "QFN prefill RSAG rank=$rank rows=2048 calls=97" "$D/native/r$rank.log"
                fi
            done
        fi
        if [[ ${NATIVE_MODE:-} == integrated-* ]]; then
            for rank in 0 1 2 3; do
                [[ $(grep -Ec "^QFN split verifier rank=$rank pos=[0-9]+ rows=4 layers=12$" "$D/native/r$rank.log") == 11 ]]
                grep -Fq '(21 of 33 drafted, 63.6%) over 11 steps' "$D/native/r$rank.out"
            done
        fi
        if [[ ${NATIVE_MODE:-} == integrated-profile ]]; then
            python3 "$B/c1.py" native "$D/native"/r{0,1,2,3}.log > /dev/null
            grep -hE '^(QFN TP prefill |QFN .*verify |QFN DN |HC prefill |Q4 profile )' "$D/native"/r?.log > "$D/instrumented-stages.txt"
        else
            python3 "$B/c1.py" native "$D/native"/r{0,1,2,3}.log > "$D/native-phase.json"
        fi
    fi
)

vllm() (
    idle
    [[ -z $(ss -H -ltn "sport = :$BENCH_PORT") ]]
    CAP=$(remaining 360); (( CAP <= 360 )) || CAP=360
    NAME=qfn-c1-$(basename "$D")
    URL=http://127.0.0.1:$BENCH_PORT
    mkdir -p "$D/vllm-work/cache/"{tmp,hf,triton,vllm,xdg}
    local -a profiler_args=() profile_cli=() reference_args=() recipe_env=()
    local context=32768
    local compilation='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,256,512,1024]}' 
    case "${REFERENCE_MODE:-graph-off}" in
        graph-off) ;;
        eager-off) reference_args=(--enforce-eager) ;;
        recipe-mtp3|eager-mtp3)
            context=262144
            compilation='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,256,512,1024]}'
            recipe_env=(-e B70_MTP_DRAFT_PREFILL_NO_PIECEWISE=1)
            reference_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"model":"/models/qwen38-w4a16-mtp-snapshot","disable_eagle_block_drop":true}')
            if [[ $REFERENCE_MODE == eager-mtp3 ]]; then reference_args+=(--enforce-eager); fi
            ;;
        *) echo "Invalid REFERENCE_MODE" >&2; exit 2 ;;
    esac
    printf '%s\n' "${REFERENCE_MODE:-graph-off}" > "$D/reference-mode"
    local batch_tokens=${REFERENCE_BATCH_TOKENS:-1024}
    if [[ ( ${PAIR_ORDER:-} == profile || ${PAIR_ORDER:-} == boundary ) && $fixture == full-chunk ]]; then batch_tokens=2048; fi
    if [[ $batch_tokens != 1024 && ( $batch_tokens != 2048 || ${REFERENCE_MODE:-} != eager-off ) ]]; then
        echo "2048 reference batch tokens require the eager attribution arm" >&2
        exit 2
    fi
    if [[ ${PAIR_ORDER:-} == profile ]]; then
        mkdir "$D/vllm-work/traces"
        timeout --kill-after=5s 60s docker run --rm --pull=never \
            --device /dev/dri:/dev/dri -e ZE_AFFINITY_MASK=0 \
            --mount "type=bind,src=$B/prove-xpu-trace.py,dst=/probe.py,readonly" \
            --mount "type=bind,src=$D/vllm-work,dst=/work" \
            --entrypoint /opt/venv/bin/python "$IMAGE" /probe.py > "$D/probe.log" 2>&1
        idle
        profiler_args=(--mount "type=bind,src=$B/qfn_profile_worker.py,dst=/work/qfn_profile_worker.py,readonly")
        profile_cli=(--worker-cls qfn_profile_worker.QFNProfileWorker --profiler-config '{"profiler":"torch","torch_profiler_dir":"/work/traces"}')
        CAP=$(remaining 360); (( CAP <= 360 )) || CAP=360
    elif [[ ${PAIR_ORDER:-} == boundary ]]; then
        mkdir "$D/vllm-work/traces"
        timeout --kill-after=5s 60s docker run --rm --pull=never \
            --device /dev/dri:/dev/dri -e ZE_AFFINITY_MASK=0 \
            --mount "type=bind,src=$B/prove-xpu-events.py,dst=/probe.py,readonly" \
            --entrypoint /opt/venv/bin/python "$IMAGE" /probe.py > "$D/probe.log" 2>&1
        idle
        profiler_args=(
            --mount "type=bind,src=$B/qfn_profile_worker.py,dst=/work/qfn_profile_worker.py,readonly"
            --mount "type=bind,src=$B/qfn_boundary_worker.py,dst=/work/qfn_boundary_worker.py,readonly")
        profile_cli=(--worker-cls qfn_boundary_worker.QFNBoundaryWorker)
        CAP=$(remaining 360); (( CAP <= 360 )) || CAP=360
    fi
    CID=$(docker run --pull=never --detach --name "$NAME" --network host --ipc host \
        --device /dev/dri:/dev/dri --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly --group-add render --group-add video \
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
        --env-file "$B/vllm-c1.env" "${profiler_args[@]}" "${recipe_env[@]}" -e PYTHONPATH=/work --mount type=bind,src=/home/haxor/models,dst=/models,readonly \
        --mount "type=bind,src=$D/vllm-work,dst=/work" --entrypoint /usr/bin/timeout "$IMAGE" \
        --signal=TERM --kill-after=8s "$CAP" vllm serve /models/qwen38-w4a16-snapshot \
        --host 127.0.0.1 --port "$BENCH_PORT" --served-model-name qwen38-flash-next-c1 \
        --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto --tensor-parallel-size 4 --enable-expert-parallel \
        --gpu-memory-utilization 0.95 --max-model-len "$context" --max-num-batched-tokens "$batch_tokens" --max-num-seqs 16 \
        --limit-mm-per-prompt '{"image":4,"video":2}' --async-scheduling --generation-config vllm \
        --compilation-config "$compilation" \
        --no-enable-prefix-caching --enable-prompt-tokens-details "${profile_cli[@]}" "${reference_args[@]}")
    trap 'docker logs "$CID" > "$D/vllm.log" 2>&1 || :; docker rm -f "$CID" >/dev/null' EXIT
    trap 'exit 143' TERM INT
    docker inspect --format '{{.Image}} {{json .Path}} {{json .Args}}' "$CID" > "$D/vllm-invocation.txt"
    ready_by=$((SECONDS + CAP - 15))
    until curl --noproxy '*' -fs --max-time 2 "$URL/health" > "$D/health"; do
        [[ $(docker inspect --format '{{.State.Running}}' "$CID") == true ]]
        (( SECONDS < ready_by ))
        sleep 1
    done
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    if [[ ${PAIR_ORDER:-} == profile ]]; then
        curl --noproxy '*' -fsS --max-time 20 -X POST "$URL/start_profile"
    fi
    curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/before.prom"
    curl --noproxy '*' --fail-with-body --silent --show-error --max-time "$(remaining 5)" -H 'Content-Type: application/json' \
        --data-binary "@$D/request.json" "$URL/v1/completions" > "$D/response.json"
    curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/after.prom"
    if [[ ${PAIR_ORDER:-} == profile ]]; then
        curl --noproxy '*' -fsS --max-time 30 -X POST "$URL/stop_profile"
        python3 "$B/trace-worker.py" "$D/vllm-work/traces" > "$D/worker-traces.json"
        exit 0
    fi
    if [[ ${PAIR_ORDER:-} == boundary ]]; then
        python3 "$B/summarize-boundaries.py" "$D/vllm-work/traces" > "$D/reference-boundaries.json"
        exit 0
    fi
    if [[ $fixture == frozen ]]; then
        if ! python3 "$B/c1.py" vllm "$D/response.json" "$D/before.prom" "$D/after.prom" \
            > "$D/vllm-phase.json" 2> "$D/vllm-score.log"; then
            cat "$D/vllm-score.log" >&2
        fi
    fi
)
case "${PAIR_ORDER:-native-first}" in
    native-first) native; vllm ;;
    reference-first) vllm; native ;;
    profile|boundary|reference-only) vllm ;;
    native-only) native ;;
    *) echo "Invalid PAIR_ORDER" >&2; exit 2 ;;
esac

idle
sudo -n journalctl -k -q --no-pager --after-cursor "$CURSOR" -o cat > "$D/kernel.log"
awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
[[ ! -s $D/xe-fault.log ]]
sha256sum -c "$SERVING"
if [[ ${PAIR_ORDER:-} == profile || ${PAIR_ORDER:-} == boundary ]]; then exit 0; fi
if [[ ${PAIR_ORDER:-} == native-only ]]; then
    case "${NATIVE_MODE:-}" in
        replay) [[ -s $D/row-replay.txt ]] ;;
        integrated-profile) [[ -s $D/instrumented-stages.txt ]] ;;
        *) [[ -s $D/native-phase.json ]] ;;
    esac
    exit
fi
if [[ ${PAIR_ORDER:-} == reference-only ]]; then [[ -s $D/vllm-phase.json ]]; exit; fi
if [[ $fixture != frozen ]]; then
    score_args=()
    if [[ $fixture == long-decode ]]; then
        score_args=(--expected-prefix "$D/fixtures/$fixture.expected-prefix.ids")
    fi
    python3 -B "$B/score-fixture.py" "$D/request.json" "$D/native" "$D" "${score_args[@]}" > "$D/comparison.json"
    exit 0
fi
for receipt in "$D/"{native,vllm}-phase.json; do [[ -s $receipt ]]; jq -e . "$receipt"; done
