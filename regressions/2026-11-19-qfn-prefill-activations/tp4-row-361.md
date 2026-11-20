# B70 TP4 parity receipt: first 512 prompt rows

This is a diagnostic, not a throughput measurement. PP4 and TP4 used the same
frozen prompt, BF16 projections, GEMV MoE, no MTP, and separate processes.
The replay stopped after the production router at zero-based layer 1. Each
layout loaded the same 512 × 10240 FP32 combined entry before PLE, and applied
its own production PLE once. The fixture checksum was
`75514512146dde52974ec8222ad5031cff582a0a1691c13aaa964489a88552b6`.

The rebuilt isolated JIT binary was
`91cfc1f3906ecaad05f985284be380b33987a2037cd562d43dd24c8906e38d94`.
All ranks exited successfully, and the native checker reported:

- Q/K/V, decay and beta matched exactly through row 361 on all ranks. The
  recurrence output and projection input at row 361 also matched exactly.
- The first differing tap was the reduced DeltaNet projection, at
  `1.56254045e-08` RMS. Adding the captured rank partials in production FP32
  order reproduced each rank's reduced output exactly.
- Both paths selected expert 492 at the cutoff, ahead of 148 by
  `0.000175952911`. This receipt does not establish 32-token parity.

The native, non-overridden activation capture traces the row-361 mismatch to
layer 0: embeddings and layer-0 entry match; layer-0 exit and layer-1 entry
both differ by `1.03291e-09` RMS. Layer-1 PLE table rows match exactly; after PLE
the residual delta is `1.12679e-09` RMS. The next recorded delta, the layer-1
mixer, is `2.32017e-05` RMS, and the expert flip makes the MoE delta
`7.43771e-04` RMS. The amplification point inside the mixer is not yet proven.

## Transport qualification

The peer-push design study in `qfn-enhanced/tp4ep4p2p.md` uses dedicated
per-source scratch and a readiness protocol. Treat the design as a reference,
not a hardware guarantee on Arc. The first Arc transport candidate should keep
FP32 payloads and rank-order summation unchanged. Check all ordered device
pairs, and verify that copy completion makes peer payloads visible before
submitting the reduce kernel. Do not borrow the study's peer atomics or wire
quantization: neither is required to test the lossless baseline.

Gate the candidate against the **same TP4** host-staged transport first (exact
collective outputs, no stale data over consecutive and differently sized
launches). Keep the frozen PP4/TP4 parity gate separate, and use unfenced,
paired engine workloads for speed measurements. Link copy measurements alone
cannot predict full-collective savings.
