# Next exclusive GPU window: combined native C1

**Staged, not launched.** Run exactly one four-rank native C1 request in a newly authorized exclusive window. This run does not launch vLLM or the separate A/B arms, change serving, or touch `retry-001`. The previous [small-M A/B then vLLM runbook](WINDOW-SMALL-MOE-AB.md) and [packed-N64 prefill comparison](PACKED-N64-PREFILL.md) remain separate experiments.

## Build and pins

The complete scratch C1 CLI and both SYCL libraries were built from source under `/tmp/qfn-catchup.pf1/combined-c1/`. `build.receipt` must be `stage=verify exit=0`. The candidate uses the pinned small-M engine source and the staged packed-N64 source, not any output from the packed-N64 collector. `native.sha256` (original baseline) and `native-candidate.sha256` (small-M-only build) remain unchanged.

`native-combined.sha256` pins these absolute files, with the same bytes as the scratch `candidate.sha256`:

```text
fa1f94ac1ccdd941578ae24a185ad234030fe07982c8798cd576c8b0b4f61694  /tmp/qfn-catchup.pf1/combined-c1/qfn-combined
55b9f1558a25632e33c91de10027c04500937cbc4f79a16047a091ffff40d445  /tmp/qfn-catchup.pf1/combined-c1/libgrimoire_gemm.so
112e8b0cca1d3e4a37996e62eb14b482106dbcafb40f6d0b078d025d7dcb73e9  /tmp/qfn-catchup.pf1/combined-c1/libgrimoire_attn.so
```

## What runs

- One request with the exact frozen 2,706 prompt IDs and 32 greedy output IDs. TP4/EP4, socket fan-out, direct peer push, E4M3 KV, and no prompt KV reuse.
- `GRIMOIRE_QFN_SMALL_MOE=1` and `GRIMOIRE_QFN_GROUPED_PACKED_N64=1`. Both switches are independent and off when unset or `0`. The CLI TP4 MTP3 diagnostic exercises the four-row small-M path; normal TP4 serving admission is unchanged. The packed switch selects N64 for the grouped 658-row prefill tail. The 2,048-row chunk keeps the existing FP32 Xe2 gate/up and down paths. The `m32` policy is unset because its current entry guard rejects verifier calls.
- Decode graph, DecodeM8, split-KV and shared merge trial are off, matching the small-M qualification recipe. No dispatch traces or fenced profiling. This is one combined configuration check, not evidence that either optimization alone improved throughput.

## Launch and pass criteria

On haxor in tmux, supply an existing empty output directory and the approved absolute Unix deadline. Allow roughly four minutes including checks: the runner refuses to start with less than 90 s plus 20 s of cleanup margin, and caps each rank at 165 s. It verifies the candidate, library resolution, baseline and input pins; acquires the exclusive four-GPU lock; and aborts if any GPU is busy. Do not stop serving to obtain the lock.

```bash
cd /home/haxor/AI/GRIMOIRE/qfn-enhanced/bench/c1-frozen
bash combined-c1.sh "$EMPTY_OUTPUT_DIR" "$APPROVED_DEADLINE_EPOCH"
```

All four ranks must exit cleanly, reproduce the frozen prompt and output IDs, agree on MTP3 steps/accepted/drafted counts, and show no xe faults or profiling markers. The throughput gate then requires **≥140 decode tok/s** AND **≥4,000 cold-prefill tok/s**, without rounding up. Rates are `31 / max(rank_decode_seconds)` and `2706 / max(rank_prefill_seconds)`. The first emitted token belongs to prefill; decode includes MTP drafting, verification, synchronization and commit. Cold means no KV reuse, not flushed page caches.

Only `stage=verify exit=0` in `receipt` is a pass. `stage=targets exit=1` means parity passed but at least one throughput target was missed; `result.json` keeps both measured rates and MTP3 counters. A failed parity stage does not publish `result.json`. Keep the raw rank, kernel and runner logs in all cases. Do not rerun after the deadline.

CPU scorer regression (no hardware queries):

```bash
bash regressions/2026-11-20-qfn-combined-c1/test-combined-score.sh
```
