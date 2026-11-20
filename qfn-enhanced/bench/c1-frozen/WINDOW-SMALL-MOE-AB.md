# Next exclusive GPU window: small-M parity, then frozen C1

**Staged; do not run without an approved exclusive four-B70 window.** Run the block below on haxor in a dedicated tmux Bash shell. Provide `APPROVED_DEADLINE_EPOCH` (absolute Unix time) and an unused `BENCH_PORT`. Budget at least 25 minutes: six small-M arms capped at 165 s each, plus startup, cleanup and a reserved 420 s for vLLM. If the window is shorter, do not start this sequence. Do not stop or replace serving or touch `retry-001`. If the GPU lock or idle check fails, abort; arrange any handover separately.

1. **Qualify the native configuration.** Run the same-binary TP4 MTP3 off/on test in `small-moe/qualification/benchmark.sh`. Require `stage=verify exit=0`, matching frozen tokens and MTP3 acceptance on all ranks, clean kernel logs, and the trace gate. Keep its off/on/on/off decode results; compare `1-on` with `1-off` and `2-on` with `2-off` when assessing the optimization. If either gate or any timed arm fails, stop. Do not substitute the off arm for a failed on arm.
2. **Compare C1 on the same workload.** Use both parity-passing native `1-on` and `2-on` samples as the two native C1 receipts: they already used one request, the pinned 2,706 prompt IDs and 32 generated IDs, fresh startup and prefix reuse off. Re-score their four rank logs with `c1.py` rather than relaunching native. Compare to one request on the existing pinned vLLM C1 recipe, also with prefix reuse off. **Native keeps MTP3 enabled exactly as qualified; the pinned vLLM C1 recipe has speculation off.** This is a comparison of engine configurations on the same workload, not a matched-speculation or kernel-only comparison. The older matched non-MTP recipe and `native.sha256` baseline pin remain unchanged and are not run here.
3. **Pass criteria.** All samples must pass those parity and cleanliness gates. **Both native on samples must meet ≥140 decode tok/s and ≥4,000 cold-prefill tok/s** without rounding up. Report each sample and the vLLM reference separately even on a target miss. Do not claim a small-M speedup or a win over vLLM solely from meeting the absolute targets. Decode is 31 tokens over first-to-last-token time; prefill is 2,706 prompt tokens over the engine's cold-prefill phase. Native uses the slowest rank per phase, vLLM one-request metric deltas. Cold means zero prompt KV reuse, not a flushed page cache. Record clocks, power caps, temperatures and host cache state without changing them.

The following block runs the two stages sequentially. Stage 1 acquires and releases its own GPU lock; stage 2 must reacquire it and recheck idle nodes. All results go to the printed `$D`. A valid comparison has `stage=verify exit=0` in `$D/receipt`. `stage=targets exit=1` means the comparison passed parity but missed at least one native target. Any other failure stage is not a qualified comparison. Do not resume a partial run after the deadline.

```bash window-runbook
(
set -euo pipefail
export LC_ALL=C
: "${APPROVED_DEADLINE_EPOCH:?}" "${BENCH_PORT:?}" "${TMUX:?}"
[[ $APPROVED_DEADLINE_EPOCH =~ ^[1-9][0-9]{9}$ && $BENCH_PORT =~ ^[1-9][0-9]{3,4}$ ]]
(( BENCH_PORT <= 65535 && BENCH_PORT != 19622 ))
B=/home/haxor/AI/GRIMOIRE/qfn-enhanced/bench/c1-frozen
S=/tmp/qfn-catchup.pf1/small-moe
Q=$S/qualification
SERVING=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-tilecut-ab-6IU2nP/serving.sha256
IMAGE=sha256:bcc91a8966e805a632f575b0467d4b8f926e1f6d7ec36cf4ecc16f3627537a05
D=$(mktemp -d /tmp/qfn-window.XXXXXXXX)
printf 'Window receipts: %s\n' "$D"
stage=preflight
container=
finish() {
    local rc=$?
    trap - EXIT
    if [[ -n $container ]]; then
        timeout 10s docker logs "$container" > "$D/vllm.log" 2>&1 || :
        timeout 15s docker rm -f "$container" >/dev/null || rc=1
    fi
    if ! sha256sum -c "$SERVING" > "$D/serving-after.sha256"; then stage=serving-check; rc=1; fi
    printf 'stage=%s exit=%d\n' "$stage" "$rc" > "$D/receipt"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
check_pins() {
    (cd "$B"; sha256sum -c inputs.sha256; sha256sum -c native.sha256; sha256sum -c native-candidate.sha256)
    (cd "$Q"; sha256sum -c runner.sha256; sha256sum -c A-B-BINARY.sha256)
    [[ $(realpath -e -- "$S/qfn-small-moe") == "$S/qfn-small-moe" ]]
    grep -Fx -f "$Q/A-B-BINARY.sha256" "$B/native-candidate.sha256" >/dev/null
    [[ $(timeout 10s docker image inspect --format '{{.Id}}' b70-flash-next:tp2) == "$IMAGE" ]]
    sha256sum -c "$SERVING"
}
idle() {
    local node occupancy rc=0
    local -a render=(/dev/dri/renderD*) nodes=(/dev/dri/by-path/pci-0000:{0a,0e,14,18}:00.0-{card,render})
    [[ ${#render[@]} == 4 ]] || return 1
    for node in "${nodes[@]}"; do [[ -c $node ]] || return 1; done
    occupancy=$(timeout 10s sudo -n fuser "${nodes[@]}" 2>&1) || rc=$?
    [[ $rc == 1 && -z $occupancy ]]
}
check_pins
python3 "$B/checkpoint-lock.py"
python3 -B "$B/test-c1.py"
cmp "$B/prompt.ids" "$Q/prompt.ids"
cmp "$B/frozen.ids" "$Q/frozen.ids"
cp "$B/"{inputs,native,native-candidate}.sha256 "$D/"
python3 "$B/c1.py" request > "$D/request.json"
CURSOR=$(sudo -n journalctl -k -q -n 1 --show-cursor --no-pager | awk '/^-- cursor:/ {print $3}')
[[ -n $CURSOR ]]
(( APPROVED_DEADLINE_EPOCH - $(date +%s) >= 1500 )) || { echo 'Need at least 25 authorized minutes'; exit 124; }

stage=small-moe
mkdir "$D/small-moe"
bash "$Q/benchmark.sh" "$D/small-moe" "$((APPROVED_DEADLINE_EPOCH - 420))"
[[ $(< "$D/small-moe/receipt") == 'stage=verify exit=0' && -s $D/small-moe/timing.tsv ]]
cmp "$Q/A-B-BINARY.sha256" "$D/small-moe/A-B-BINARY.sha256"
for arm in 1-on 2-on; do
    grep -Fxq 'GRIMOIRE_QFN_SMALL_MOE=1' "$D/small-moe/$arm/options.env"
    python3 "$B/c1.py" native "$D/small-moe/$arm"/r{0,1,2,3}.log > "$D/native-$arm.json"
done

stage=vllm-preflight
exec 9>/tmp/grimoire-qfn-gpus.lock
flock -n 9
idle
listeners=$(ss -H -ltn "sport = :$BENCH_PORT")
[[ -z $listeners ]]
(( APPROVED_DEADLINE_EPOCH - $(date +%s) >= 420 )) || { echo 'vLLM reserve exhausted'; exit 124; }
mkdir -p "$D/vllm-work/cache/"{tmp,hf,triton,vllm,xdg}
URL=http://127.0.0.1:$BENCH_PORT
container=qfn-window-$(basename "$D")
VLLM_CUTOFF=$(($(date +%s) + 350))
request_budget() {
    local left=$((VLLM_CUTOFF - $(date +%s)))
    (( left >= $1 )) || { echo 'vLLM request budget exhausted' >&2; return 124; }
    printf '%s\n' "$left"
}

stage=vllm-start
cid=$(timeout --kill-after=2s 20s docker run --pull=never --detach --name "$container" --network host --ipc host \
    --device /dev/dri:/dev/dri --group-add render --group-add video \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    --env-file "$B/vllm-c1.env" --mount type=bind,src=/home/haxor/models,dst=/models,readonly \
    --mount "type=bind,src=$D/vllm-work,dst=/work" --entrypoint /usr/bin/timeout "$IMAGE" \
    --signal=TERM --kill-after=8s 360s vllm serve /models/qwen38-w4a16-snapshot \
    --host 127.0.0.1 --port "$BENCH_PORT" --served-model-name qwen38-flash-next-c1 \
    --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto --tensor-parallel-size 4 --enable-expert-parallel \
    --gpu-memory-utilization 0.95 --max-model-len 32768 --max-num-batched-tokens 1024 --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image":4,"video":2}' --async-scheduling --generation-config vllm \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,256,512,1024]}' \
    --no-enable-prefix-caching --enable-prompt-tokens-details)
timeout 10s docker inspect --format '{{.Image}} {{json .Path}} {{json .Args}}' "$cid" > "$D/vllm-invocation.txt"
cap=$(request_budget 10)
timeout "$cap" curl --noproxy '*' -fsS --max-time 2 --retry-all-errors --retry 600 --retry-delay 1 \
    --retry-max-time "$cap" "$URL/health" > "$D/health"

stage=vllm-request
request_budget 15 >/dev/null
curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/before.prom"
cap=$(request_budget 10)
curl --noproxy '*' -fsS --max-time "$((cap - 5))" -H 'Content-Type: application/json' \
    --data-binary "@$D/request.json" "$URL/v1/completions" > "$D/response.json"
curl --noproxy '*' -fsS --max-time 5 "$URL/metrics" > "$D/after.prom"
python3 "$B/c1.py" vllm "$D/response.json" "$D/before.prom" "$D/after.prom" > "$D/vllm.json"

stage=vllm-cleanup
timeout 10s docker logs "$cid" > "$D/vllm.log" 2>&1
timeout 15s docker rm -f "$cid" >/dev/null
container=
idle
sudo -n journalctl -k -q --no-pager --after-cursor "$CURSOR" -o cat > "$D/kernel.log"
awk '/xe.*(Fault response|Timedout job|GPU HANG|Engine memory CAT error|wedged)/' "$D/kernel.log" > "$D/xe-fault.log"
[[ ! -s $D/xe-fault.log ]]
check_pins

stage=targets
jq -s --argjson decode 140 --argjson prefill 4000 '
    def targets: (.decode_tok_s >= $decode and .cold_prefill_tok_s >= $prefill);
    {targets:{decode_tok_s:$decode,cold_prefill_tok_s:$prefill},
     native_1_on:(.[0] + {targets_pass:(.[0]|targets)}),
     native_2_on:(.[1] + {targets_pass:(.[1]|targets)}),
     vllm:.[2],
     native_targets_pass:(.[0]|targets) and (.[1]|targets)}
' "$D/native-1-on.json" "$D/native-2-on.json" "$D/vllm.json" > "$D/comparison.json"
jq . "$D/comparison.json"
jq -e '.native_targets_pass' "$D/comparison.json" >/dev/null
stage=verify
)
```
