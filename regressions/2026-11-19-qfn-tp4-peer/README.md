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
