# TP4 peer-push vs host staging on four Arc Pro B70s

Run `bash run.sh /path/to/isolated/grimoire /path/to/empty-output-dir` in tmux
with all four GPUs enumerated. Wait for `<output-dir-basename>-done` with a
bounded `tmux wait-for` (or read `receipt` if it already exists). The runner
acquires `/tmp/grimoire-qfn-gpus.lock`, checks GPU occupancy, and bounds each
rank with TERM only. It requires neither a checkpoint nor a serving instance.

The regression calls the production `peer_copy_setup` and `tp_allreduce_sum`
once with host staging and once with required peer-push, over the same ranked
sockets. It compares every output byte on every rank for the sequence of row
counts `512, 146, 146, 512` (width 2,560). The payload changes between calls to
catch stale receive slots. One lane in 17 uses `2^24, 1, -2^24, 1` across the
ranks: the required sequential FP32 sum is 1, not a regrouped sum of 0 or 2.
The test also checks that the peer path actually advances its sequence, so
host fallback cannot make the peer arm pass.

Production enables this only with `GRIMOIRE_P2P=1` (or `required`) on four
Megatron ranks. Rank 0 brokers dedicated sockets, not payloads. Each rank's
copy engine pushes its FP32 partial into a slot on each peer. Copy completion
precedes the readiness exchange; a second exchange prevents a sender from
reusing a slot until its receiver has consumed it. Each GPU adds the four
partials locally in rank order. `GRIMOIRE_P2P_CHUNK_MB` must agree across ranks.
The default and `auto` retain the host path for TP4; PP and TP2 are unchanged.

The test uses 8 MiB slots by default so the full prefill slice can be pushed
in one operation. Set `QFN_TEST_PEER_CHUNK_MB=4` to verify the same payload
split across two operations. On haxor, the pre-change binary completed all
four host references per rank, then failed `peer_copy_setup` on every rank.
The opt-in implementation matched host staging on all 16 rank/call checks
with each slot size (`diagnostics/act-peer-uHytkg/{red,green,split4}`).

The test qualifies this collective on these payloads, not end-to-end speed or
PP4/TP4 model parity. The frozen prompt's independent token-19 mismatch stays
open.

## Frozen model workload

Set `QFN_MODEL` to the dense text snapshot and `QFN_PLE_SOURCE` to the
original snapshot. Build the host checker with `make bin/qfn_parity_receipt`,
then run `bash benchmark.sh ISOLATED_JIT FROZEN_PROMPT_IDS EMPTY_OUTPUT_DIR`
in tmux. Like the collective test, this runner uses the four-GPU lock and
writes a receipt. It pins 512-row chunks, BF16 projections, GEMV, MTP off,
no prefix cache, and 8 MiB peer slots (one push per destination for a 5 MiB
prefill reduction). Startup and model loading are outside the timed phases.

`qfn_parity_receipt --tp4` uses logits and token IDs from all ranks for both
transport arms. If all 32 IDs agree, the runner times host→peer and peer→host
pairs with the diagnostic logs disabled. It verifies the timed token IDs
still match the gate and writes raw spans and derived throughputs to
`timing.tsv`. Prefill is 2,706 tokens to prompt-ready. Decode is the 31
subsequent tokens, from first token to last. Each rate uses the longest
observed span across the four ranks for that phase.

Set `QFN_BENCH_MODE=grouped` to keep peer transport fixed and compare dense
XMX + routed GEMV (`dense`) against dense XMX + routed grouped XMX (`grouped`).
The grouped gate checks that every rank used the grouped kernel in every
chunk, and compares all 32 greedy IDs before timing. Use `QFN_BENCH_PAIRS=1`
for an initial screening pair; the default uses both execution orders.

Set `QFN_BENCH_MODE=profile` for fenced grouped-TP4 diagnostics. `profile.tsv`
separates collective calls (including peer waits but excluding the preceding
queue drain), route-plan and routed-device time, and the remaining layer-loop
work. `QFN_PROFILE_DETAIL=1` adds the existing Q4 stage breakdown. None of
these diagnostic samples are throughput measurements. `QFN_BENCH_MODE=hc`
compares unfused vs. fused residual-add/RMSNorm with grouped experts and
peer-push fixed; it verifies the fused path ran and all 32 IDs match before
running a trace-off pair.

`QFN_BENCH_MODE=mtp-preflight` is currently a **negative** qualification
check for `GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3` under host-staged TP4. It passes
only when every rank reports the speculation-off loader refusal. No tokens
or timings from this mode count as an MTP result.
