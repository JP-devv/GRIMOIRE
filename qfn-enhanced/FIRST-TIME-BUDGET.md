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
