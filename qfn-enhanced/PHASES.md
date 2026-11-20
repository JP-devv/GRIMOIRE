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
| E — cold prefill | **Current implementation: rank-local grouped W4A16 XMX prefill.** Add a regression for stable compaction of each rank's resident expert window, then test the device gate/up/down/gather with original top-k weights and stored codes. Measure dense XMX alone before adding grouped experts so each gain has its own receipt. Keep both candidates opt-in until their arithmetic and frozen quality gates pass. |
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
Phase E candidate is grouped W4A16 XMX on TP4's 128 resident experts per rank
(vLLM/radiance both batch routed experts; Grimoire's existing TP4 branch
still uses per-row GEMV). Reuse the existing native grouped kernel and compare
with dense-XMX-only on the same binary before considering fabric refinements.
The ~19.5 ms host decode step is a different lever: qualify MTP3 acceptance
and verification, not a claim that transport can triple single-token decode.

## Current receipts

- Isolated worktree: `/home/haxor/AI/GRIMOIRE-qfn-qgate`, branch `qfn-b70-tp4-parity`.
- Transport regression: `diagnostics/act-peer-uHytkg/{red,green,split4}`.
- Frozen host vs peer TP4: `diagnostics/act-peer-ab-B1CWCv/`; trace-off paired benchmark recorded in `timing.tsv`.
- The serving binary stays frozen. Serialize GPU tests with `/tmp/grimoire-qfn-gpus.lock` and use a bounded tmux receipt wait, never duplicate a live job. Reboot/power-cycle only as recovery for a confirmed wedge, not a benchmark timeout.
