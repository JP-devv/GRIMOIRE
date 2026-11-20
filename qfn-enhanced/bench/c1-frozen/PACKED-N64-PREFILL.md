# Independent packed-N64 prefill qualification

Staged only. This run is separate from the small-M/MTP3 decode run and does not touch `retry-001` or serving. It consumes the existing collector's `build-collection/retry-001/` receipt and `qualification/A-B-BINARY.sha256`; it never runs `prepare`, relinks, or restarts anything. If the successful receipt and verified prepared binary are missing, stop. A standalone device fixture or an intermediate full-build binary is not a substitute for the pinned A/B artifact.

## What changes

Use the same `qfn-packed-n64` executable, same TP4/EP4 placement, and same frozen 2,706 prompt IDs → 32 output IDs in both arms. `GRIMOIRE_QFN_GROUPED_PACKED_N64=0` is the existing FP32 Xe2 full-chunk/grouped-tail control; `=1` replaces only the grouped gate/up and down projections for the 658-row tail. Both keep the 2,048-row Xe2 FP32 `m32` path. **Decode stays fixed:** MTP off, DecodeM8 on, decode graph off, same TP4 peers/fan-out. The small-M decode candidate and the Xe2 tail experiment are not enabled. Both start with no prompt KV reuse; do not flush the OS cache.

The existing packed runner first gates both arms on all four ranks: frozen IDs, finite and identical shard logits, intended 2,048-row/658-row dispatches, and the same DecodeM8 path. It rejects xe faults. Then it times **control → packed → packed → control** with no trace or profiling, checking token parity again. `comparison.json` reports **cold-prefill tok/s = 2,706,000 / max(rank prefill_ms)** for each arm and the packed change against the same-build control in each pair. Decode tok/s is a sanity check, not the optimization metric.

The report also gives the existing C1 native baseline from the pinned, successful `verifier/gate-ZwV33FAa/1-mtp-off` archive. Its existing `native.sha256` pin is unchanged. This archived value is historical context, not a controlled estimate of the packed-N64 effect: use the fresh same-build control for that conclusion.

## CPU-only checks

```bash
cd /home/haxor/AI/GRIMOIRE/qfn-enhanced/bench/c1-frozen
python3 -B regressions/2026-11-20-packed-n64-prefill/test-packed-prefill.py
bash packed-prefill.sh check
```

`check` verifies the workload, input pins, baseline archive and serving fingerprint, and prints the historical C1 phase rates. Exit 3 means the collector has not published a usable candidate; do not retry in a loop. The comparison stays blocked until the existing collector finishes successfully.

## Next authorized window, unlaunched

From tmux, with an empty output directory and at least 12 authorized minutes left (six arms capped at 100 s plus cleanup), run:

```bash
cd /home/haxor/AI/GRIMOIRE/qfn-enhanced/bench/c1-frozen
bash packed-prefill.sh run "$EMPTY_OUTPUT_DIR" "$APPROVED_DEADLINE_EPOCH"
```

The original packed runner owns the exclusive GPU lock, checks idle devices, and bounds each rank by the window deadline. If the lock, checksums, parity or timing checks fail, stop; do not stop any serving process or rerun the collector. Use `comparison.json` only with `stage=verify exit=0` in the outer `receipt`. Per-arm evidence is in `arms/`; `packed-ab.log` records the original runner output.
