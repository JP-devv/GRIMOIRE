# Four-row QFN verifier attention: profile and split-KV trial

This is an isolated SYCL/Level Zero build. It does not replace the serving binary or touch either checkout. `build.receipt` and `split-build.receipt` record the full verifier CLI links; `candidate.sha256` and `split-candidate.sha256` identify the exact executables. The packed N64 A/B is independent.

## What the prior 44.8 ms actually measured

The QFN-TP4 MTP3 diagnostic dispatches through `prefill_qwen4_exp`. Its dense attention calls `launch_dflash2_block_attention` with the default `split_keys=false`. At four verifier rows this sends six work-groups per rank, one per query head, through the entire key range. There is no separate final reduction on that path. The fenced `full-attn` bucket additionally includes other attention work such as projections, writes, and waits. The `launch_flash_decode_batched` path in generic batch verification cannot by itself explain this QFN result because it was not selected.

`GRIMOIRE_VERIFY_ATTN_PROFILE=1` requests SYCL device timestamps. For the existing block path it records the fused kernel event, `splits=1`, `reduce_ns=0`, K dim-major, V token-major, and E4M3, per rank, layer, and round. Events are collected outside the layer loop. When the batched path is selected, it times its actual split-KV scan and final reduction separately on the same device. It logs the split count passed to the launch, not a heuristic estimate. That optional diagnostic waits for completion before reading both events; use neither profile for throughput. Subtracting the summed kernel device time from the existing fenced wall-time bucket gives a residual, not another measured kernel duration.

## One isolated speed candidate

`GRIMOIRE_QFN_VERIFY_SPLIT_KV=1` uses the existing batched split-KV SYCL kernels for the four-row dense TP4 diagnostic shape (6Q/1KV, head dim 256, context >= 2048, dim-major K, in-order queue). It maps the sampled shape to 32 split work-groups per rank and reuses the verifier partials. The default remains block. Shape, capacity and explicit exact-verifier guards prevent this experiment from changing unsupported paths. Combining partials changes floating-point accumulation order; do not use it in serving given only a build pass.

The target budget is much tighter than attention alone. The archived 32-token run had 31 decode tokens over 11 steps, or about 2.82 per step; 140 tok/s needs roughly **20 ms/step** at comparable acceptance. The prior fenced step was about 101.5 ms; even removing all 44.8 ms of fenced full attention leaves roughly 57 ms. After qualifying split-KV, profile small-row routed MoE (24.5 ms previously) and then reduce what's left in the attention projections, HC, DeltaNet, and cross-rank sync rather than chasing the 2.6 ms local vocabulary head first. For prefill, 2,706 tokens at 4,000 tok/s allows 676.5 ms versus the observed roughly 2.5 s; the packed N64 test attacks the largest localized tail cost (378 ms), but reaching the budget also requires full-chunk and other-phase reductions. None of these are measured gains yet.

## Staged commands, not run

Only after a fresh GPU window is authorized, run one of these from tmux using its actual absolute cutoff and a different empty directory for each invocation:

```sh
cd /tmp/qfn-catchup.pf1/verify-attn-profile
bash benchmark.sh "$BASELINE_PROFILE_DIR" "$APPROVED_DEADLINE_EPOCH" profile
bash benchmark.sh "$SPLIT_PROFILE_DIR" "$APPROVED_DEADLINE_EPOCH" split-profile
bash benchmark.sh "$SPLIT_AB_DIR" "$APPROVED_DEADLINE_EPOCH" split-ab
```

Profiles publish `attention.tsv` only after all four ranks match the frozen prompt and 32 greedy IDs and supply 12 attention events per round, with no xe fault. Split profiles list scan and reduction time in separate columns. The A/B produces no device profiling: it checks the actual split dispatch on four ranks in two gate arms, then runs control → split → split → control without trace logging, checking frozen IDs again on every timed arm and publishing `timing.tsv` only on success. For a first window, prioritize the baseline profile and split parity. The runner neither stops serving nor proceeds if a B70 node is occupied.
