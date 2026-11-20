# timemap — where the time goes, per phase, per engine

Written 2026-11-20 on `haxor` from measurements taken the same day. Every number
below is **measured**, tagged with the artifact it came from; derived numbers are
marked *(derived)*. Nothing here is quoted from a document unless it says so.

Two engines, one model (Qwen3.8-Flash-Next), same box, same four Arc Pro B70s:

| | native | vLLM lane |
| --- | --- | --- |
| engine | GRIMOIRE, C++/SYCL, TP4 whole-expert EP4 | vLLM 0.30.0 + 44 patches + L3, TP4+EP4, MTP k=3 |
| weights | W4A16 + FP8 PLE from NVMe | W4A16 + FP8 PLE in place |
| artifact | `diagnostics/act-xe2-profile-0zms3D` | live lane, `switch.py`, measured this session |

---

## 1. How these numbers were obtained

**Native.** One run with every profiler enabled: `GRIMOIRE_Q4_TIMING=1`,
`GRIMOIRE_HC_TIMING=1`, `GRIMOIRE_QFN_TP_PREFILL_REDUCE_TIMING=1`,
`GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1`, `GRIMOIRE_QFN_DN_PROFILE=1`, frozen
2,706-token prompt, chunk 2048, 32 greedy IDs. The five per-chunk phase lines sum
**exactly** to the span (18.5 + 470.3 + 341.7 + 276.9 + 414.8 = 1,522.2 =
span 1,522.166), so this is a partition, not overlapping buckets. Decode lines are
`synced` per token. Reduction buckets are fenced (`(fenced)` in the marker).

**vLLM.** The lane was booted with `switch.py --start` and stopped afterwards;
`tp4-local.env` and `serve-tp4.args` were sha256-verified restored. Step counts
come from `vllm:spec_decode_num_drafts_total` deltas, tokens from `usage`, decode
time from first-token to last-token over SSE.

### Methodology traps (all of these cost a wrong number first)

1. **SSE chunks are steps, not tokens** under spec decode — one chunk carries 1–4
   tokens. Counting chunks as tokens reported 44.5 tok/s where the truth was 101.
2. **`--speculative-config` exists twice**: `args/serve-tp4.args:20` and
   `EXTRA_ARGS`. Stripping one left MTP on; the fake "no-MTP" arm reproduced
   22.99 ms/step to the hundredth, which is what exposed it.
3. **The prefix cache makes TTFT useless** for prefill: 125 ms warm vs 709 ms
   cold on the same 2,853-token prompt.
4. **A `trap` restore using a relative path after a `cd` silently fails.** Restores
   must hash-verify; one env file was left modified until that was added.
5. **vLLM device-event hooks arm in the API frontend (pid 1), not the worker** that
   runs the model, so a single trace file is empty or overwritten. Per-PID files
   were also empty.
6. **Stale `launcher.lock` blocks the next boot** and the engine never comes up;
   the symptom is a connection refused after a "launched" line.

---

## 2. Native prefill — 2,048-row chunk

Span **1,522.166 ms** (`QFN TP prefill profile rank=0 pos=0 rows=2048`).

| phase | ms | share | what it is |
| --- | ---: | ---: | --- |
| HC | **470.3** | 30.9 % | hyper-connection projections + 5 surrounding passes, 48 layers |
| MoE | **414.8** | 27.3 % | routed experts + its all-reduce, 48 layers |
| DeltaNet | **341.7** | 22.4 % | 36 linear-attention layers incl. their reduce |
| full-attn | **276.9** | 18.2 % | 12 QSA layers incl. their reduce |
| PLE | **18.5** | 1.2 % | n-gram table, 32,768 rows read in 1.6 ms |

### HC, decomposed (sums to 470.3)

| pass | ms | note |
| --- | ---: | --- |
| down | 246.3 | one BF16 GEMM per layer; **52 % of all HC time** |
| up | 92.2 | second BF16 GEMM |
| combine | 43.3 | |
| mean | 37.3 | |
| norm | 33.0 | |
| inject | 16.7 | |
| silu | 1.5 | |

### MoE, decomposed (from the fenced detail line, 2,048 chunk)

| part | ms |
| --- | ---: |
| all-reduce | 185.9 |
| routed device (experts) | 144.0 |
| router + topk | 23.2 |
| remainder | 61.7 |

### Transport inside the span: 365.2 ms, by stage

| stage | MoE | DeltaNet | attention | PLE | total | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| copy | 126.5 | 96.1 | 31.7 | 2.9 | **257.2** | 70 % |
| ready | 31.6 | 15.5 | 5.3 | 0.4 | **53.0** | 15 % |
| sum | 14.8 | 10.9 | 3.5 | 0.3 | **29.5** | 8 % |
| reuse | 12.8 | 9.2 | 3.1 | 0.3 | **25.4** | 7 % |
| drain | 0.02 | 0.02 | 0.01 | 0.3 | **0.34** | — |

Copy is the payload: `elems=5,242,880` × 4 B = **20.97 MB per call**, × 97 calls =
**2.03 GB per chunk per rank in 257 ms ≈ 8 GB/s** *(derived)*. That is the PCIe
P2P rate on this box, not a kernel defect. **The transport problem is bytes and
peer arrival, not the sum kernel** (sum is 8 %).

---

## 3. Native prefill — 658-row tail

Span **1,076.964 ms** (`pos=2048 rows=658`), for 32 % of the rows of the chunk
above but 71 % of its time.

| phase | ms | note |
| --- | ---: | --- |
| MoE | **556.9** | routed device 378.1 — the Xe2 policy is gated `MR == 2048` |
| HC | 238.8 | down 153.5, up 37.8, combine 14.4, mean 14.1, norm 11.4, inject 6.6, silu 1.0 |
| full-attn | 133.9 | |
| DeltaNet | 131.5 | |
| PLE | 15.8 | |

Transport in the tail is **178.5 ms** with a different signature: copy 40.5,
ready **69.5** (dominant — waiting on peers), sum 4.9, reuse 4.8 for MoE alone.
The tail is latency-dominated where the big chunk is bandwidth-dominated.

---

## 4. Native decode — per token

31 tokens, synced, wall **21.45 ms/token**; the phase lines sum to **21.19**.

| phase | ms | share | calls/step | exchange ms | drain ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE | **8.11** | 38 % | 48 | 1.970 | 3.850 |
| HC | **5.66** | 27 % | — | — | — |
| DeltaNet | **4.42** | 21 % | 36 | 1.386 | 2.387 |
| full-attn | **2.01** | 9 % | 12 | 0.472 | 1.249 |
| head | 0.68 | 3 % | — | — | — |
| PLE | 0.31 | 1 % | 1 | 0.030 | 0.008 |
| **total** | **21.19** | | **97** | **3.858** | **7.494** |

Call composition: **48 MoE + 36 DeltaNet + 12 attention + 1 PLE = 97 per step**.
This resolves the "97 = 2×48+1" from earlier: it is one mixer reduction and one
MoE reduction for each of 48 layers — 36 GDN, 12 full-attn — plus PLE.

HC per token: `hc_down_inject` **2.152** and `hc_up_mean` **2.122** are fenced and
account for 4.27 of the 5.66 ms.

### Reduction sub-buckets, per step (run totals ÷ 31)

| kind | calls/step | copy | ready | sum | reuse | drain | exchange |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE | 48 | 0.626 | 0.635 | 0.440 | 0.269 | 3.850 | **1.970** |
| DeltaNet | 36 | 0.470 | 0.366 | 0.324 | 0.226 | 2.387 | **1.386** |
| attention | 12 | 0.156 | 0.130 | 0.109 | 0.077 | 1.249 | **0.472** |
| PLE | 1 | 0.013 | 0.004 | 0.009 | 0.004 | 0.008 | **0.030** |

Payload: `10,240 B` per call × 97 = **993 KB/step**, moved in 3.858 ms =
**≈257 MB/s** *(derived)*. The exchange is latency, not bandwidth.

**Two conclusions.** Real exchange is **3.86 ms/token (18 %)** — that is all that
any transport work can ever recover at this shape. **Drain is 7.50 ms/token
(35 %)**, and drain is not transport: it is the compute queue that must finish
before the exchange starts, so the routed MoE GEMM per step is only ≈2.3 ms while
≈6.1 ms of "MoE time" is compute serialized behind reductions.

---

## 5. vLLM lane — what is measured, what is blocked

### Measured (live, this session)

| variant | rows/step | ms/step | tok/step | tok/s | artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| MTP3, graph | 4 | **22.80** (p50 22.54) | 2.31 | 101.3 | live probe |
| MTP3, repeat | 4 | 22.47 | 2.55 | 113 | live probe |
| MTP3, `B70_MOE_DECODE_OP=1` | 4 | 22.22 | 2.31 | 103.9 | live probe |
| **no MTP** | 1 | **17.87** (p50 17.83) | 1.00 | 56.0 | live probe |
| MTP3, `--enforce-eager` | 4 | 22.4 | 2.4 | 107 | live probe |

Facts that fall out:

- **Marginal row = 1.64 ms** *(derived: (22.80 − 17.87)/3)* — 9 % of a one-row step.
- **Step time is invariant to acceptance.** 1.31 and 1.55 accepted/step both gave
  22.5–22.8 ms. Acceptance changes tokens out, never round cost.
- **Graph replay buys nothing here**: eager 22.4 vs graph 22.8 ms (the lane's own
  env notes call `--enforce-eager` "flat single-stream", consistent).
- `B70_MOE_DECODE_OP=1` moved the step by −0.58 ms and **produced no engine log
  marker**, so it is **not attributed**.

### Blocked: kernel-level attribution

The instrument for this was never built. `kernels/b70-step-trace/` holds
`b70_step_trace.sycl`, `build.sh`, `vllm-step-trace.patch` and a self-validating
test — **`libb70_step_trace.so` does not exist anywhere on the box**. Four
cheaper routes attempted this session and their failure modes:

| route | outcome |
| --- | --- |
| `Module._call_impl` + XPU events (works in a test container) | armed in the **frontend** pid 1 only; worker never traced |
| per-PID trace files + periodic flush | no files produced |
| instrument on `/work` + PYTHONPATH | collided with a stale `launcher.lock`; engine never came up |
| `VLLM_TORCH_PROFILER_DIR` + `/start_profile` (eager) | **404** — endpoints not registered in this build |

Next concrete step: build that `.so` with an ABI-matched oneAPI toolchain (the
documented `COMPILE_IMAGE` no longer exists and the runtime image has no `icpx`),
then mark mixers, MoE, HC, PLE, draft and verify. Expected cost ≈1.5–2 ms/step.

---

## 6. Side by side

| | native | vLLM (MTP3) | vLLM (no MTP) |
| --- | ---: | ---: | ---: |
| ms per step | **18.8** (current stack) / 21.45 (this artifact) | 22.80 | 17.87 |
| rows per step | 1 | 4 | 1 |
| tokens per step | 1.00 | 2.31–2.95 | 1.00 |
| decode tok/s | 53.22 | 101–131 | 56.0 |

**Per step the native engine is at parity or better (18.8–21.5 vs 22.8). The entire
2.3× decode gap is tokens per step.** No amount of kernel work closes it: at 1.0
token/step, 140 tok/s needs a **7.1 ms** step, 2.6× faster than the lane itself.

Speculation arithmetic *(derived)*, using the lane's measured row scaling:
a round of `18.8 + 4.9 = 23.7 ms` yields **97 tok/s at 2.31 acceptance** and
**124 tok/s at 2.95**. **140 tok/s needs a round ≤ 21.1 ms at 2.95 accepted.**
The native verify round measures **92 ms** — the gap to 23.7 ms is ≈68 ms of
non-batched work in the verifier, and that is the largest single item left.

---

## 7. What each phase means for the targets

- **Prefill 4,000 tok/s** needs 676.5 ms for 2,706 tokens; this artifact's chunks
  cost 1,522 + 1,077 = 2,599 ms. Largest blocks: HC 709 (470+239), MoE 972
  (415+557), transport 543. HC's two GEMMs run far under their arithmetic floor
  (down+up = 338 ms for work that should take ~17 ms), and the tail's MoE 557 ms is
  a dispatch problem (Xe2 gated at `MR == 2048`), not arithmetic.
- **Decode 140 tok/s** is a speculation problem, not a phase problem: the phases
  sum to 21.19 ms and the exchange is only 3.86 of it. Fixing every transport item
  buys 3.86 ms/step; fixing the verifier buys 68 ms/round.
- **Transport** is worth at most 3.86 ms/token (18 %) in decode and 365 ms of
  1,522 (24 %) in prefill, where 70 % of it is the 2.03 GB of payload copy. Any
  further transport work must move fewer bytes or overlap more, not shave latency.

---

## 8. Appendix — verbatim markers (rank 0, `profile-m8`)

```
QFN TP prefill profile rank=0 pos=0 rows=2048 calls=97 grouped=48 span_ms=1522.166 \
  communication_ms=365.214 routed_plan_ms=4.602 routed_device_ms=144.020 rest_ms=1008.329
QFN TP prefill profile rank=0 pos=2048 rows=658 calls=97 grouped=48 span_ms=1076.964 \
  communication_ms=178.528 routed_plan_ms=2.212 routed_device_ms=378.061 rest_ms=518.163
qwen4_exp prefill M=2048 (sub-chunks of 2048): PLE 18.5  HC 470.3  DeltaNet 341.7  full-attn 276.9  MoE 414.8 ms  (PLE rows read 32768 in 1.6 ms)
qwen4_exp prefill M=658 (sub-chunks of 658): PLE 15.8  HC 238.8  DeltaNet 131.5  full-attn 133.9  MoE 556.9 ms  (PLE rows read 43296 in 11.4 ms)
qwen4_exp prefill HC M=2048: combine 43.3  norm 33.0  down 246.3  silu 1.5  up 92.2  mean 37.3  inject 16.7 ms
qwen4_exp prefill HC M=658: combine 14.4  norm 11.4  down 153.5  silu 1.0  up 37.8  mean 14.1  inject 6.6 ms
qwen4_exp prefill M=2048 MoE detail (fenced): router+topk 23.2 ... | all-reduce: mixer 175.6  MoE 185.9 ms
qwen4_exp decode per token (31 tokens, synced): PLE 0.31  HC 5.66  DeltaNet 4.42  full-attn 2.01  MoE 8.11  head 0.68 ms
QFN TP decode profile rank=0 tokens=31 routed_ms=0.000 ple_ms=9.569 hc_ms=175.316 deltanet_ms=136.947 attention_ms=62.367 moe_ms=251.394 head_ms=21.016
QFN TP decode projections rank=0 tokens=31 int4_gu_ms=0.000 int4_down_ms=0.000 hc_down_inject_ms=66.721 hc_up_mean_ms=65.779 (fenced)
QFN TP decode reduce rank=0 kind=moe calls=1488 drain_ms=119.358 copy_ms=19.394 ready_ms=19.692 sum_ms=13.641 reuse_ms=8.335
QFN TP decode reduce rank=0 kind=deltanet calls=1116 drain_ms=73.980 copy_ms=14.558 ready_ms=11.347 sum_ms=10.047 reuse_ms=7.016
QFN TP decode reduce rank=0 kind=attention calls=372 drain_ms=38.713 copy_ms=4.838 ready_ms=4.021 sum_ms=3.388 reuse_ms=2.377
QFN TP decode reduce rank=0 kind=ple calls=31 drain_ms=0.245 copy_ms=0.398 ready_ms=0.130 sum_ms=0.276 reuse_ms=0.139
QFN TP decode payload rank=0 kind=moe calls=1488 bytes=15237120 min_bytes=10240 max_bytes=10240
```

Reproduce the native side with
`regressions/2026-11-20-qfn-xe2-gate-up/run-candidate.sh` in `profile` mode with
`GRIMOIRE_QFN_DN_PROFILE=1` in the environment; the vLLM side with
`switch.py --start qwen38-flash-next` plus the probe described in §1 (the lane's
own numbers come from `vllm:spec_decode_*` deltas and `usage`).
