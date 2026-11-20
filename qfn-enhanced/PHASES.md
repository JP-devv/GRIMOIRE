# Flash-Next on four Arc Pro B70s — phase ledger

This ledger tracks implementation against [proposition.md](proposition.md),
not the vLLM deployment. Targets: **4,000 cold prefill tok/s** and **140
single-stream decode tok/s**. On the frozen 2,706-token prompt the prefill
budget is 676.5 ms. Keep native C++/SYCL in the build, runtime and tests.
The vLLM and radiance materials are design references, not measurements of
Grimoire or requirements to reproduce either engine's software stack.

| Phase | Status and next implementation gate |
|---|---|
| A — fixtures and receipts | Frozen prompt and separate dense/PLE/draft model roles; native readers. Preserve the prompt hash, 512-row split, and separate prefill/decode spans. Add a longer stream and representative workloads before comparing against BetterBench. |
| B/C — storage, sharding, correctness floor | Native dense snapshot + external PLE; resident W4A16 TP4. Keep the independent PP4-vs-TP4 token-19 mismatch open (`364` vs `15060`). Identical-entry replay fixed row-361 routing in isolation, but replacing layer-zero projection alone did not fix the end-to-end mismatch. Do not change router scores, stored codes, or rank order to hide it. |
| D — decode and MTP3 | Next isolate native decode launch/collective overhead versus expert reads. Use size-aware transport only after paired measurements. Continue draft loader/HC fixture gates, then measure native MTP3 exact greedy parity, accepted tokens per step, ms/step, and useful tok/s. At 2.94 tokens per step, 140 tok/s requires at most 21.0 ms/step; acceptance is measured, not assumed. |
| E — cold prefill | **Implemented: opt-in rank-local grouped W4A16 XMX prefill.** Host regression covers stable compaction of the four resident windows, missing slots, and invalid IDs. Frozen TP4 A/B confirms dispatch on every rank/chunk and all 32 IDs agree. Profile complete: mixer/other dominates; a fused HC residual-add/RMSNorm screening pair did not establish a useful win. Next measure HC down/up GEMMs separately and qualify a projection candidate before AOT and longer-prompt tests. |
| F — transport and expert placement | Four-rank Level Zero peer-push is opt-in (`a261fb8`), with original FP32 bytes and rank-order sum. 5 MiB and 146-row reductions match socket staging on every rank, including a split transfer. Same-TP4 model gate: 32/32 IDs match and top logit/margin match. Keep whole-expert placement fixed while measuring transport. Width-sliced experts and device-side signaling are separate candidates, not prerequisites to this baseline. |
| G — exact reuse | Not timed as a speedup on the cold workload. Resume equivalence and slot isolation are prerequisites. |
| H — K8/V4 and QSA | Keep BF16 KV on the transport baseline. Qualify K8/V4 independently for 6Q/1KV per rank, accuracy and speed. QSA is a separate checkpoint view, not a shortcut to the dense target. |

## F — confirmed transport measurement (frozen 2,706 input / 32 output)

The same isolated TP4 binary passed the 32-ID comparison. At position 2,706,
both transports chose `760`, with top logit `29.1116219` and margin
`11.862854`. Diagnostic-independent timing used paired host→peer and
peer→host order, with only phase timing enabled (31 decode tokens):

| Pair | Host prefill tok/s | Peer prefill tok/s | Host decode tok/s | Peer decode tok/s |
|---|---:|---:|---:|---:|
| 1 | 133.90 | 141.26 | 51.42 | 46.20 |
| 2 | 133.72 | 141.29 | 51.33 | 45.89 |

Peer-push saved about 1.1 s in prefill but increased short decode time about
11–12%. Do not promote it unconditionally. Its remaining ~19.15 s prefill is
28× the 676.5 ms budget, so transport alone cannot close the gap. The selected
Phase E change is grouped W4A16 XMX on TP4's 128 resident experts per rank.
This applies the vLLM prefill approach of batching routed rows per expert,
using the existing native kernel. The radiance fabric work informs Phase F
separately; its AMD collective timings are not Intel kernel measurements.
The ~19.5 ms host decode step is a different lever: qualify MTP3 acceptance
and verification, not a claim that transport can triple single-token decode.

## E — grouped prefill screening pair (same TP4 JIT binary and peer transport)

| Arm | Prefill ms | Prefill tok/s | Decode ms (31 tokens) | Decode tok/s |
|---|---:|---:|---:|---:|
| Dense XMX + routed GEMV | 10,420.845 | 259.67 | 683.765 | 45.34 |
| Dense XMX + routed grouped XMX | 5,288.845 | 511.64 | 668.904 | 46.34 |

Grouped prefill was 1.97× faster in this pair; do not infer a decode gain
from one pair. The grouped branch ran on every rank in all six chunks. All
32 generated IDs match both the dense-XMX arm and the original TP4 peer
baseline. Top logit is **not** bit-exact (`29.1345291` → `29.071537`), so
this is a frozen greedy-token gate, not a general numerical-quality pass.
The PP4 token-19 mismatch is still open independently. Reaching 4,000
prefill tok/s still requires cutting 5.29 s to at most 676.5 ms. The next E
measurement must split this remainder before choosing a kernel or transport
rewrite. For D, compare exact greedy MTP3 verification against host-staged
target-only decode, and record both step latency and accepted tokens/step.

## E/F — grouped TP4 prefill profile

Ported `GRIMOIRE_QFN_TP_PREFILL_REDUCE_TIMING=1` to the common reduction call
sites (PLE, DeltaNet, full attention, MoE) so the opt-in peer path is timed too.
A queue drain before the collective is excluded from communication and counted
where it belongs in the rest of the span. Communication includes waiting for
peer ranks, not just fabric traffic. Three exclusive buckets are reported over
the layer loop; startup, prompt setup before the loop and the vocabulary head
are outside it. Each rank performed 582 reductions (97 per chunk).

| Rank | Layer-loop ms | Communication ms | Routed experts ms (plan + device) | Rest ms |
|---|---:|---:|---:|---:|
| 0 | 5253.45 | 758.17 | 1628.81 | 2866.48 |
| 1 | 5253.19 | 727.16 | 1688.16 | 2837.87 |
| 2 | 5252.39 | 675.51 | 1740.17 | 2836.71 |
| 3 | 5257.00 | 645.78 | 1771.33 | 2839.90 |

The remainder is largest and includes shared MLP/router and host work in
addition to the mixer. A separate, more heavily fenced `GRIMOIRE_Q4_TIMING=1`
pass makes hyper-connections the largest named cost inside that remainder:
rank 3 recorded **1,414.4 ms HC**, 45.2 ms PLE, 706.6 ms DeltaNet and 493.8 ms
full attention (the last two include their mixer reductions). Both profiles
kept the original grouped TP4 32-ID sequence. Use the original 5.29 s trace-off
measurement as the benchmark, not either diagnostic span.

Selected experiment: the vLLM `b70-hc` reference motivates fusing the HC
epilogues, but its trial was not a shipping vLLM path. Reuse Grimoire's own
SYCL residual-add/RMSNorm fusion for the pending-residual prefill cases; keep
both BF16 GEMMs, inject, and output gate unchanged. Test it against unfused HC
with grouped W4A16 and peer transport fixed. Timing starts only after the
fused path executes every chunk and matches all 32 greedy IDs.

| Same-binary trace-off pair | Prefill ms | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Unfused HC | 5293.865 | 511.16 | 46.15 |
| Fused residual-add/RMSNorm | 5266.895 | 513.78 | 45.99 |

The fused arm ran 94 times per chunk on rank 3 and passed the 32-ID gate;
its top logit changed (`29.071537` → `29.1903648`). The 27 ms delta is only
0.51% in one pair. Leave it opt-in; this does not establish a useful speedup.
Next use `GRIMOIRE_HC_TIMING=1` to separate HC projection cost from epilogues
before choosing a same-precision projection change. The PP4 token-19 issue
remains a separate qualification gate.

## D — TP4 MTP3 gate blocked at load

Ran all four ranks on host staging with the same frozen IDs, `GRIMOIRE_MTP=1`
and `GRIMOIRE_MTP_K=3`. Each refused `… and speculation off` at load (exit 1).
No MTP-on token sequence, verifier steps or timings exist to compare; the
host-staged MTP-off result remains 51.33–51.42 tok/s (1 token/target step,
19.45–19.48 ms/step for this 31-token decode window).

Do not delete the refusal alone: the external draft packed INT4 reader is
not wired to MtpHead4, and the current main-checkpoint drafter assumes
replicated experts. Wire the separate dense + packed draft snapshot, qualify
its rank-local placement and verifier/commit state on TP4, then compare all
32 IDs. A passing run must count actual emitted decode tokens and completed
verifier rounds, excluding the prefill anchor and flagging any fallback or
terminal truncation; use trace-off wall latency, not fenced verify time.

## Current receipts

- Isolated worktree: `/home/haxor/AI/GRIMOIRE-qfn-qgate`, branch `qfn-b70-tp4-parity`.
- Transport regression: `diagnostics/act-peer-uHytkg/{red,green,split4}`.
- Frozen host vs peer TP4: `diagnostics/act-peer-ab-B1CWCv/`; `timing.tsv`, `parity.log`, `receipt`.
- Frozen dense vs grouped screening: `diagnostics/act-xmx-ab-N8CMZj/`; `timing.tsv`, `parity.log`, `receipt`. Build and host checks: `diagnostics/act-xmx-NVEZ9l/`.
- Grouped profile: `diagnostics/act-xmx-profile-NxRRs7/`; detailed HC/mixer profile: `diagnostics/act-xmx-detail-73gSwU/`.
- HC fusion screening: `diagnostics/act-xmx-hc-R1DT0S/`; `timing.tsv`, `parity.log`, `receipt`. Build: `diagnostics/act-xmx-hc-build-pj7IZI/`.
- TP4 MTP3 refusal: `diagnostics/act-mtp3-refusal-blQRsZ/`; `run.log` and `receipt`.
- The serving binary stays frozen. Serialize GPU tests with `/tmp/grimoire-qfn-gpus.lock` and use a bounded tmux receipt wait, never duplicate a live job. Reboot/power-cycle only as recovery for a confirmed wedge, not a benchmark timeout.
