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
