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
python3 "$B/c1.py" request > "$D/request.json"
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

finish() {
    local rc=$?
    trap - EXIT
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
(
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
    export GRIMOIRE_PROMPT_IDS="$(< "$B/prompt.ids")" GRIMOIRE_PRINT_IDS=1 GRIMOIRE_GENERATION_PHASES=1
    CAP=$(remaining 90); (( CAP <= 165 )) || CAP=165
    SOCK=$(mktemp -d /tmp/qfn-c1-sock.XXXXXXXX)
    pids=(); trap 'for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done; wait || :; rm -f -- "$SOCK/p"; rmdir -- "$SOCK"' EXIT
    trap 'exit 143' TERM INT
    mkdir "$D/native"
    for rank in 3 2 1 0; do
        GRIMOIRE_TP_RANK=$rank GRIMOIRE_TP_SOCKET=$SOCK/p \
            timeout --signal=TERM --kill-after=4s "$CAP" /tmp/qfn-catchup.pf1/verifier/qfn-verifier \
            -m /home/haxor/models/qwen38-w4a16-snapshot --proj bf16 --ctx 32768 -p frozen -n 32 \
            > "$D/native/r$rank.out" 2> "$D/native/r$rank.log" &
        pids[rank]=$!
    done
    for rank in 0 1 2 3; do wait "${pids[rank]}"; unset 'pids[rank]'; done
    for f in "$D/native"/r?.out; do
        grep -Fxq '  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$f"
        grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$f"
    done
    python3 "$B/c1.py" native "$D/native"/r{0,1,2,3}.log > "$D/native-phase.json"
)
(
    idle
    [[ -z $(ss -H -ltn "sport = :$BENCH_PORT") ]]
    CAP=$(remaining 360)
    NAME=qfn-c1-$(basename "$D")
    URL=http://127.0.0.1:$BENCH_PORT
    mkdir -p "$D/vllm-work/cache/"{tmp,hf,triton,vllm,xdg}
    CID=$(docker run --pull=never --detach --name "$NAME" --network host --ipc host \
        --device /dev/dri:/dev/dri --group-add render --group-add video \
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
        --env-file "$B/vllm-c1.env" --mount type=bind,src=/home/haxor/models,dst=/models,readonly \
        --mount "type=bind,src=$D/vllm-work,dst=/work" --entrypoint /usr/bin/timeout "$IMAGE" \
        --signal=TERM --kill-after=8s "$CAP" vllm serve /models/qwen38-w4a16-snapshot \
        --host 127.0.0.1 --port "$BENCH_PORT" --served-model-name qwen38-flash-next-c1 \
        --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto --tensor-parallel-size 4 --enable-expert-parallel \
        --gpu-memory-utilization 0.95 --max-model-len 32768 --max-num-batched-tokens 1024 --max-num-seqs 16 \
        --limit-mm-per-prompt '{"image":4,"video":2}' --async-scheduling --generation-config vllm \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,256,512,1024]}' \
        --no-enable-prefix-caching --enable-prompt-tokens-details)
    trap 'docker logs "$CID" > "$D/vllm.log" 2>&1 || :; docker rm -f "$CID" >/dev/null' EXIT
    trap 'exit 143' TERM INT
    docker inspect --format '{{.Image}} {{json .Path}} {{json .Args}}' "$CID" > "$D/vllm-invocation.txt"
    timeout "$CAP" curl --noproxy '*' --fail --silent --show-error --max-time 2 --retry-all-errors \
        --retry 600 --retry-delay 1 --retry-max-time "$CAP" "$URL/health" > "$D/health"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/before.prom"
    curl --noproxy '*' -fsS --max-time "$(remaining 5)" -H 'Content-Type: application/json' \
        --data-binary "@$D/request.json" "$URL/v1/completions" > "$D/response.json"
    curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/after.prom"
    python3 "$B/c1.py" vllm "$D/response.json" "$D/before.prom" "$D/after.prom" > "$D/vllm-phase.json"
)
idle
sudo -n journalctl -k -q --no-pager --after-cursor "$CURSOR" -o cat > "$D/kernel.log"
awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
[[ ! -s $D/xe-fault.log ]]
sha256sum -c "$SERVING"
for receipt in "$D/"{native,vllm}-phase.json; do [[ -s $receipt ]]; jq -e . "$receipt"; done
