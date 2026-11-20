# QFN cold-prefill cost and HC-down M64 candidate

## What packed N64 can affect

The frozen 2,706-token C1 reference is **2,571.436 ms (1,052.330 tok/s)** using the slowest rank in `/tmp/qfn-catchup.pf1/verifier/gate-ZwV33FAa/1-mtp-off`. The 4,000 tok/s limit is **676.500 ms**; the gap is **1,894.936 ms**.

At `/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-profile-0zms3D/profile-m8`, all four `matched-r*.ids` match `bench/c1-frozen/frozen.ids`, and the parent receipt is `stage=verify exit=0`. This is a separate fenced run (2,676.136 ms slowest-rank wall), not an additive decomposition of the unfenced C1 reference. All numbers below are from the same rank 0 unless noted:

| Chunk | Layer span | Reduce communication | Routed plan | Routed device | Residual |
|---|---:|---:|---:|---:|---:|
| 2,048 | 1,522.166 ms | 365.214 ms | 4.602 ms | 144.020 ms | 1,008.329 ms |
| 658 | 1,076.964 ms | 178.528 ms | 2.212 ms | 378.061 ms | 518.163 ms |

The 2,048-row chunk uses Xe2 projections; packed N64 changes gate/up and down in the **658-row grouped tail**. The full routed-device bucket there is **378.061–417.904 ms** across ranks. It includes work the switch does not change, and the archive does not time the two projections separately. Communication wait can also move between ranks as one arrives sooner; do not add rank maxima or subtract phase and transport totals twice.

Subtracting even 417.904 ms from the 2,571.436 ms unfenced reference as an *optimistic transfer of the profile budget* leaves **2,153.532 ms / 1,256.54 tok/s**, still **1,477.032 ms** over budget. No isolated packed N64 speedup is established, and this cross-run estimate is not a strict bound on a new measurement.

## Largest remaining phase

Rank 0 in `profile-m8`: hyper-connections (HC, the model's residual-stream mixer) **709.1 ms**, including **399.8 ms** for down (BF16 activation staging + projection); collectives **543.742 ms** across layers. The inclusive attention and DeltaNet buckets already contain their respective collectives. That profile used the untiled HC path. As a cross-check, `diagnostics/act-xe2-m8-combo-ab-p4GQLV/profile-combo` also matches all 32 frozen IDs on all ranks and has a passing parent receipt. With HC tiling on, it still records HC **654.2 ms** including down **325.3 ms** (231.1 + 94.2), and **568.507 ms** collective communication. This is a different stack, so the two HC measurements are not a controlled before/after. Down is the largest individual HC operation in both.

## Candidate: `GRIMOIRE_QFN_HC_DOWN_M64=1`

In TP4 ranked XMX prefill only, for the eligible BF16 10,240 → 320 HC down projection, use 64×64 workgroups instead of 256×64. Independent 32×64 subgroup tiles still traverse K in order and write FP32; there is no split-K reduction. The 658-row chunk now exposes 55 workgroups instead of 15, and the 2,048-row chunk 160 instead of 40. Pack exactly 320 columns in the existing BF16-pair layout rather than clearing a 512-column buffer before each projection. The scratch requirement for this projection falls from 10 MiB to 6.25 MiB; the allocator may retain a larger buffer for other projections. Arithmetic and activation rounding are intended to match the legacy tiled path. No change to decode, other GEMM tilings, or collective ordering is intended.

Default off. Retain `GRIMOIRE_QFN_HC_DOWN_TILED=1` on both sides when isolating the schedule change. The CPU regression covers exact flag selection, the baseline workgroup map, full and partial output-tile coverage, and compact packing across the last BF16 pair and N tile. The full native CLI and both local SYCL libraries built; `build.receipt` is `stage=verify exit=0`. `candidate.sha256` pins the deployed tuple. The build does not validate device results or speed.

## Next GPU window (not run)

With a fresh exclusive reservation, compare the same pinned candidate, M64 off vs on, holding every other feature fixed (including packed N64). First gate both arms on the four ranks' 32 frozen IDs and identical MTP3 steps/accepted/drafted counts; verify HC dispatch for both 2,048/658-row chunks with `GRIMOIRE_BUDGET_DEBUG` and the tiled hit counter. Use diagnostic HC timings only for attribution. Once both gates pass, remove diagnostic flags and run off/on/on/off with cold requests. Recheck parity and the device journal on each result; score slowest-rank phase times. Report each sample's exact pass/fail against 4,000 prefill and 140 decode tok/s, independently of whether M64 improves on the same-build control.
