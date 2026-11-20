# QFN paired C1 benchmark — frozen 2,706 → 32

**Staged only; no new benchmark has run.** Use the commands below only in a newly authorized, exclusive four-B70 window. This is an engine-phase comparison, not a production-service cutover or a reproduction of the historic MTP3 leaderboard number.

| Field | Frozen value |
|---|---|
| Request | One in flight (C1), greedy, exactly 2,706 prompt IDs, no chat template or retokenization, 32 emitted IDs |
| Fixture | `bench/c1-frozen/prompt.ids` and `frozen.ids`; both hashed in `inputs.sha256` |
| Native | `/tmp/qfn-catchup.pf1/verifier/qfn-verifier` plus its two libraries (`native.sha256`); TP4, EP4, peers and socket fan-out, E4M3 KV, 2,048 + 658 prefill; MTP and prefix reuse off |
| vLLM | Installed `b70-flash-next:tp2`, image ID `sha256:bcc91a8966e805a632f575b0467d4b8f926e1f6d7ec36cf4ecc16f3627537a05`; run it with TP4 + EP4 despite the tag, BF16 KV (`auto` with BF16 model), 1,024 batch-token limit, async scheduler, **no speculative-config** and no prefix cache |
| Snapshot roles on haxor | W4A16 target `/home/haxor/models/qwen38-w4a16-snapshot`, FP8 PLE `/home/haxor/models/qwen38-w4a16`, INT4 draft `/home/haxor/models/qwen38-w4a16-mtp-snapshot` (pinned but **not loaded** in this pair) |
| Snapshot identity | `checkpoint-lock.json` freezes configs, indexes, tokenizer files and shard realpaths, sizes, mtimes and safetensors-header hashes. The verifier reads no tensor payload; this is metadata identity, not a byte-for-byte checksum of all weights. The role snapshots share underlying shards. |

Both engines use local models, all experts resident, 32,768 context capacity and the same four PCI devices. The vLLM arm uses the installed image and TP4 recipe's FP8-PLE/oneCCL setup in a dedicated container, not the production launcher. It removes MTP, selects power-of-two graph captures, bounds context and disables prefix caching. Do not call this the deployed *configuration*. Native E4M3 vs vLLM BF16 KV and different chunk sizes are deliberate implementation differences; do not imply numerically equivalent kernels. Compare MTP3 as a separate matched experiment, not by enabling it on one side here.

## Rates and acceptance

- **Cold prefill tok/s** = `2706 / prefill_seconds`. Native uses the maximum of the four `prefill_ms` values ÷ 1,000 (`prompt_ready - start`). vLLM uses the one-request delta of `vllm:request_prefill_time_seconds_sum` (first scheduled to first token in the local vLLM source). These are each engine's internal phase boundary, not HTTP TTFT. Compare as an engine-phase proxy; don't subtract the two as exact kernel time.
- **Decode tok/s** = `(32 - 1) / decode_seconds`. Native uses the maximum of four `decode_ms` values ÷ 1,000. vLLM uses `vllm:request_decode_time_seconds_sum` delta. Both intervals run from first to last token; the first token is not counted again. Do not divide by SSE chunks, model-load time or total request time.
- **Cold** means zero prompt KV hits, not that Linux page caches are flushed. Native starts fresh with `GRIMOIRE_PREFIX_CACHE=0`; vLLM starts with `--no-enable-prefix-caching` and the reply must report `cached_tokens=0`. Record PLE/OS-cache state and GPU clocks/caps/temperatures from the window, but do not flush or adjust them. Keep fresh engine startup per sample, exclude model loading/graph capture, never interleave requests, and reverse the engine order on a repeat.
- All four native rank logs and the vLLM reply must match the pinned IDs and lengths. Require successful rank exits, no xe faults, one histogram-count increment per vLLM phase, zero cached tokens, and no fenced profiling. Keep raw logs even if an arm is rejected; do not publish it as a parity-passing pair. The scorer is fail-closed if the installed image does not expose the expected token IDs or metric series. Record HTTP latency separately if needed; never substitute it for these phase times.

## CPU-only validation (no inference)

```bash
cd /home/haxor/AI/GRIMOIRE/qfn-enhanced/bench/c1-frozen
sha256sum -c inputs.sha256
sha256sum -c native.sha256
python3 checkpoint-lock.py
python3 -B test-c1.py
python3 c1.py request > /tmp/qfn-c1-request.json
```

## Commands for the next authorized window — unlaunched

Run on haxor in a dedicated tmux Bash shell. The operator supplies `APPROVED_DEADLINE_EPOCH` and an unused `BENCH_PORT`. Do not stop or replace an occupied lane from this runbook; abort if not idle and arrange any handover/restore separately. Save the serving fingerprint and kernel journal cursor before the pair and recheck after it. Do not resume a partial pair after the deadline.

```bash
set -euo pipefail
Q=/home/haxor/AI/GRIMOIRE/qfn-enhanced
B=$Q/bench/c1-frozen
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
D=$(mktemp -d /tmp/qfn-c1.XXXXXXXX)
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
```

### Native arm

```bash
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
```

### vLLM arm (dedicated loopback endpoint)

Use the image's `serve --help` in the authorized window if the cold-cache flag is rejected; do not silently retry with caching on or pull a different image. The isolated container uses only the pinned `vllm-c1.env`, its own cache volume, and read-only model mounts. `timeout` *inside* the container protects the deadline even if the client disconnects.

```bash
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
```

After both commands and cleanup, check idle GPUs and the pair's receipts:

```bash
idle
sudo -n journalctl -k -q --no-pager --after-cursor "$CURSOR" -o cat > "$D/kernel.log"
awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
[[ ! -s $D/xe-fault.log ]]
sha256sum -c "$SERVING"
for receipt in "$D/"{native,vllm}-phase.json; do [[ -s $receipt ]]; jq -e . "$receipt"; done
```
