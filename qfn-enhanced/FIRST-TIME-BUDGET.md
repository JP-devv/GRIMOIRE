# Frozen C1 starting budget

The archived unfenced MTP-off native receipt records 2,571.436 ms for prefill and 48.92 generated tok/s. The 4,000 tok/s ceiling is 676.5 ms for 2,706 IDs, leaving 1,894.936 ms to remove. At three emitted tokens per speculative round, 140 tok/s allows 21.428571 ms per round. Fresh paired timings and all reference worker phases are unmeasured.

The diagnostic attention A/B changed prefill latency by +7.21% and −8.14% in reverse order. Its spread does not establish a repeatable end-to-end win.

| Stage | Archived native 2,048 rows, rank 0 (ms) | Reference | Observed native work and deciding comparison |
|---|---:|---|---|
| HC | 470.3 | Unmeasured | Down projection alone is 246.3 ms with BF16 staging. Compare device times and bytes for packing plus GEMM against the actual worker layout before choosing persistent packing or fusion. |
| Dense attention | 276.9 | Unmeasured | Dense attention stages 16-key tiles with barriers. Compare cache preparation plus attention on 6 Q heads, 1 KV head and width 256. |
| DeltaNet | 341.7 | Unmeasured | Includes reduction. Time QKV, Z and output projections apart from convolution and recurrence on worker shapes. |
| MoE | 414.8 | Unmeasured | Includes 144.020 ms routed-device work and 185.9 ms reduction. Compare packing, projections, activation and gather jointly with the reference dispatch and actual routes. |
| PLE | 18.5 | Unmeasured | Includes embedding lookup and reduction. Compare cold KV with equivalent warm OS and PLE row-cache state. |
| Collectives | 365.214 | Unmeasured | Nested in the rows above. Copy was 257.2 ms with 20,971,520 logical payload bytes per reduction. Compare arrival, explicit copy and reduction intervals and count actual transmitted bytes per link. |

These are fenced phase counters from the archived profile. The rank-0 chunk span is 1,522.166 ms. The counters are inclusive and cannot be summed into that span or credited as throughput savings. A fresh profile must capture the same stage boundaries on all ranks and keep queue drain and rank arrival skew separate from payload transport.

## New four-rank evidence

The split-KV diagnostic reduced the two unprofiled decode phases from 1,064.267 and 1,074.475 ms to 606.359 and 613.485 ms. Identical frozen IDs and 21/33 acceptance held in all arms. The saving was 41.628 and 41.908 ms per completed round. It leaves 384.930–392.056 ms to remove over 11 rounds at the same emission count. Split-KV already uses one KV workgroup per split for all 24 queries. Coalescing the existing token-major V staging and reusing FP32-input batched GEMV for the sharded head are bounded follow-on experiments. The archived head bucket is about 2.57 ms, which bounds the opportunity there.

Target-only direct verifier replay used the same proposed rows at a fixed context and reversed the order within one process start. The maximum per-rank durations for M1/M4/M4/M1 were 82.320207/98.672462/89.239902/71.938025 ms. Each predicted token matched a sequential replay. The measured increments were 16.352255 and 17.301877 ms. This uses the ordinary dense block verifier and excludes draft cache warming and commit. It does not qualify speculative serving state.

The full-chunk HC activation fusion fired 94 times per rank and avoided a separate conversion pass reading 7,885,291,520 logical activation bytes per rank. It increased the frozen prefill by 8.155 ms and 52.628 ms against the same-build controls in reversed order. All four arms matched the frozen IDs. Reject this fusion. The separate resident pack prototype targets repeated weight packing while preserving BF16 values and FP32 reduction order. It charges 600 MiB per rank and must demonstrate the same context capacity before any serving promotion.

FP32 Level Zero reduce-scatter/all-gather is a separate transport prototype for the 2,048 × 2,560 full-chunk payload. It aims to move 31,457,280 outgoing peer bytes per rank per reduction instead of 62,914,560, with five shard groups instead of twenty copy chunks at a 1 MiB peer capacity. It adds a local gather copy. The archived full-chunk copy bucket was 257.2 ms. Expected whole-chunk savings are roughly 90–145 ms if the byte reduction survives readiness and local copies. Test canonical cancellation and exact payload bytes against the host sum before model A/B. The two-stage transfer keeps canonical FP32 addition order and does not change KV or wire dtype.

## Recovered reference worker trace

All four XPU model workers recorded a first scheduler step of exactly 2,048 rows with the installed TP4/EP4 image in eager mode. The standalone trace proof captured a known device operation. The worker export timed out after 30 s and left incomplete gzip files. The complete CPU stage events were recovered without a second trace launch. These host boundary spans include submission, profiling overhead and waits. Device kernel times, collective transport and payload bytes remain unmeasured in this export.

| Stage | Reference worker host union, min–max across rank (ms) | What to compare next |
|---|---:|---|
| HC | 97.537–99.405 | Same activation dtype, including the VNNI pack or tiled launcher costs |
| Dense attention | 29.579–30.355 | FlashAttention 2 and native dense block ownership, cache write and packing |
| DeltaNet | 78.839–86.978 | Official reference GDN path, projections, layout and recurrence |
| MoE | 117.862–124.453 | Full rank-local route, projections and communication |
| PLE layer | 1.930–2.861 | Keep per-layer work separate from pre-forward preparation |
| PLE pre-forward | 7.221–90.499 | Rank skew, host index preparation, stream wait and row gather |
| Collective transport | Unmeasured | Record actual logical call shapes and separate queue spans from bytes on the wire |

Whole worker host span was 548.168–553.419 ms. These numbers come from a separate instrumented run and do not enter the parity score. The event-based worker fallback captures smaller current-stream intervals without a large Kineto export.

## Bounded reference worker device events

The smaller event-based fallback passed the known-operation check and then captured all four workers at 2,048 rows and at the first one-row decode step. The returned raw input had exactly 2,048 IDs and no cached prompt tokens. The reference image selected FlashAttention 2, XPU WNA16 MoE, official GDN, TP4/EP4 and FP8 PLE. Installed reference KV is BF16 and native KV is E4M3. The stateful PLE pre-forward and first decode step have different synchronization from the full-chunk layer stages.

| Stage | Native rank 0 fenced full chunk (ms) | Reference four-rank current-stream union (ms) | Deciding experiment |
|---|---:|---:|---|
| HC | 470.3 | 93.150–94.601 | Resident exact VNNI down operands, then isolate down/up projections against worker shapes |
| Dense attention | 276.9 | 19.291–19.405 | Whole-stage dense attention with cache preparation at identical causal boundaries |
| DeltaNet | 341.7 | 61.459–62.080 | Native QKV and Z/out projection vs convolution and recurrence boundary timings |
| MoE | 414.8 | 161.238–162.126 | Compare routed-device path and collective separately at identical routing and live expert rows |
| PLE | 18.5 | 2.168–2.198 layer; 7.726–8.104 prepare | Compare host preparation, H2D staging and gather at matching row-cache state |
| Collective boundary | 365.214 | 68.355–82.748 | Native full-chunk payload experiment and reference XCCL transport attribution |

Reference whole-step spans were 397.821–400.703 ms, with up to 3.016 ms of host entry skew. The first one-row instrumented step had up to 11.282 ms of host entry skew and is heavily perturbed by instrumentation and first-use work. The reference actually submitted 1,022,361,600 logical bytes per rank through the hooked all-reduce API for the 2,048-row step, from 97 BF16 2,048 × 2,560 reductions and one additional 5,242,880-byte call. The 97 full-width native FP32 reductions account for 2,034,237,440 logical input bytes per rank. Logical API bytes and requested peer-copy bytes are different measurements. Physical wire bytes and transport-only reference time are unmeasured. Current-stream intervals may include waits on asynchronous work, and collective boundaries are nested within phase spans. Do not add them to the table or credit these profiled times as serving throughput gains.

## Integrated RSAG and split-KV diagnosis

All 97 full-chunk reductions used the new transport and all 12 dense layers used split attention on each of the 11 verifier rounds. The profile is separate from the provisional unprofiled integration sample.

| Stage | Fresh integrated native, rank 0, 2,048 rows (ms) | Earlier reference worker stream range (ms) | Observed next work |
|---|---:|---:|---|
| HC | 326.5 | 93.150–94.601 | Up was 148.3 ms, standalone gated mean 37.0 ms, down 70.2 ms. The reference logs deterministic K-split at width 2,048, with packing at load time. Fuse native up and mean first without changing the FP32 branch sum. |
| Dense attention | 254.4 | 19.291–19.405 | Native staged keys beyond each query tile's maximum causal position. On the initial 2,048 rows, skipping those positions removes 8,128 of 16,384 staging iterations per head and layer without changing the legal keys or softmax order. Expect roughly 40–100 ms of full-chunk savings if this loop dominates the stage. |
| DeltaNet | 276.6 | 61.459–62.080 | Native QKV 68.805 ms, convolution/normalization 45.204 ms, scan 12.142 ms, Z 41.768 ms and reduction 66.172 ms. Target projection and staging work before replacing the recurrence. |
| MoE | 318.9 | 161.238–162.126 | Native routed work 135.633 ms, transport boundary 98.206 ms. The separate 658-row tail still spends 559.2 ms in inclusive MoE, so it needs a native short-chunk operator after full-width work. |
| PLE | 16.1 | 2.168–2.198 layer; 7.726–8.104 prepare | Keep host preparation and model gather distinct. |
| Collective boundary | 187.999 | 68.355–82.748 | Native RSAG copied 3,051,356,160 requested outgoing peer bytes per rank. Rank 0 copy time was 129.360 ms, local gather 9.795 ms. These boundaries include readiness and are nested in the phases above. |

The native fenced full-chunk span was 1,192.519 ms. Native and reference instrumentation have different overhead and synchronization. Absolute target budget for the initial 2,048 rows would be 512 ms at 4,000 tok/s. The 658-row suffix needs comparable work removal and has a separate 164.5 ms budget at that rate.

The integrated verifier's fenced step was about 59.98 ms, with draft 4.17 ms and verify 55.24 ms. Full attention is now roughly 4.2–4.8 ms per verify step, while MoE is roughly 24–25 ms, HC about 6 ms, and DeltaNet about 7 ms. With 31 decode tokens over 11 rounds, 140 tok/s permits 20.12987 ms per round. Eliminating the entire remaining attention bucket would leave the round far above budget. The next decode implementation should prioritize the complete routed stage, including routing and returns, sharing committed row state and traffic across the fixed four-row schedule. Head batching, coalesced V staging, and recent scratch reuse are smaller controlled experiments. Estimate up to 1–2 ms per round for head reuse and V staging separately, bounded by their measured stage spans. Record actual cold and reused allocations in the scratch trial rather than assigning an additive gain.

The HC up/mean candidate packs the four branches together while keeping BF16 input rounding and FP32 accumulation. It avoids roughly 16.1 GB of separate gate writes and reads per rank over 96 mixers at 2,048 rows. Budget 35–70 ms as a hypothesis against the 185.3 ms measured up-plus-mean path. The packed weights cost 600 MiB per rank and full context capacity needs qualification. Neither this estimate nor the causal tile estimate is a serving result.

## Combined native serving candidate, final diagnostic profile

The combined causal-tile, HC up/mean and FP32 RSAG build passed the frozen 32-token output gate in the separate profiling run. This table uses all four ranks. Native fenced and reference current-stream scopes differ.

| Stage | Native 2,048 rows, rank range (ms) | Reference 2,048 rows, rank range (ms) |
|---|---:|---:|
| HC | 168.9–170.4 | 93.150–94.601 |
| Dense attention | 204.4–204.5 | 19.291–19.405 |
| DeltaNet | 333.8–333.9 | 61.459–62.080 |
| MoE | 317.8–318.0 | 161.238–162.126 |
| PLE | 16.3 | 2.168–2.198 layer; 7.726–8.104 prepare |
| Inclusive collective boundary | 183.293–187.790 | 68.355–82.748 |
| Whole chunk span | 1041.505–1042.758 | 397.821–400.703 |

The combined HC up-plus-mean path recorded 31.8 ms on rank 0. It uses persistent packed BF16 weights, the existing BF16 activation conversion and an in-kernel FP32 mean. The stage profile alone does not give the serving gain. Fresh full-chunk work is dominated by DeltaNet projections and movement, then the inclusive MoE stage and dense attention. The irregular 658-row chunk spans 937.877–938.783 ms and its inclusive MoE phase is 560.9–561.0 ms. Routed-device work there is 381.825–420.629 ms. Collective spans overlap the phase allocations above, and reference wire-only time is still unmeasured.

The remaining full-width gap calls for stage-specific changes to the QFN forward path. Prioritize the DeltaNet projection layouts and exact dense cache handling before changing recurrence math. Keep the known four-rank placement to avoid full-width intermediates where consumers can use a shard. For irregular chunks, replay live expert rows and measure the complete route/pack/projection/gather sequence before integrating the existing tail Xe2 work. The decode diagnostic still needs a batched routed stage that retains target-only recurrent, convolution, KV and PLE continuation. Its small-M alternative failed the frozen output gate. The authoritative performance measurements remain the unprofiled ABBA phase receipts.
