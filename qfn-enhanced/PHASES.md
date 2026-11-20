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
| F — transport and expert placement | Four-rank Level Zero peer-push is opt-in (`a261fb8`), with original FP32 bytes and rank-order sum. 5 MiB and 146-row reductions match socket staging on every rank, including a split transfer. Same-TP4 model gate: 32/32 IDs match and top logit/margin match. The Xe2 FP32 baseline already matches vLLM's contiguous whole-expert EP4 placement; loaded shapes and full routed-output replay passed on all ranks. OneCCL on the current stack is inside noise overall and slower in decode in both orders, so socket fan-out stays the transport baseline. Width-sliced experts and device-side signaling are separate candidates, not prerequisites to this baseline. |
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

The four-rank red gate was recaptured with `GRIMOIRE_QFN_MTP_SOURCE` pointing
at the real draft snapshot: `diagnostics/act-mtp3-red-cZXanC/receipt` is
`stage=mtp-gate exit=1`; each rank log records the same load refusal.

The isolated code now imports the native packed group-32 INT4 reader and
prepares `MtpHead4` from separately indexed dense BF16 tensors and packed
expert bytes with unchanged BF16 scales and zero point 8. The TP4 draft
layout assigns 128 routed experts per rank and the shared expert to rank
zero, then reduces the result. The active TP4 refusal is unchanged pending
device qualification. The speculative greedy loop now checks agreement on
the start position, accepted row count (including the anchor), and verifier
correction before committing state. It rejects a declined verifier rather
than calling it a successful TP4 commit.

The native host control-flow test derives the next IDs from greedy replay,
injects a three-proposal sequence with proposal two deliberately wrong, and
checks all four simulated ranks' logical KV length, recurrent/convolution
state and next logits against non-speculative replay. It does **not** qualify
GPU state: a live deterministic injection from the frozen 2,706-token
MTP-off output and per-rank device state/logit comparison are still required
before lifting the refusal. The host-staged frozen fixture is derived from
all four agreeing rank receipts: anchor `760`, injected proposals
`[1156, 16562, 16563]`, verifier correction `16561`, next greedy `864`.
The expected logical KV length is 2708 after committing the anchor and first
proposal, 2709 after replaying the correction. Host red/green and the fixture
are in `diagnostics/act-mtp3-wire-Z8TEtR/` (`model-red.log`, `host.log`,
`host.exit=0`, `frozen-rejection-fixture.txt`). A marker recheck found
`build.exit=0` and `host.exit=0`; no compiler failure to repair. The
loader's `STDtype::BF16` matches the enum in `include/b70/safetensors.hpp`.
A forced rebuild of `bin/test_generation` and rerun both passed without
warnings (`test-generation-recheck-build.log`, `test-generation-recheck.log`,
`test-generation-recheck.exit=0`), including four modeled ranks rejecting
proposal two and matching replay state/logits. This is not the live GPU
state receipt.

**Frozen off/on gate on the exact existing build.** `act-mtp3-wire-Z8TEtR/`
recorded `build.exit=0` and `host.exit=0`; the host partial-rejection test
and packed expert read reported PASS. The lock-protected paired attempt
`diagnostics/act-mtp3-paired-TCvMzJ/` exercised grouped TP4 peer push on
both arms with MTP depth 3 requested only for the second. MTP off matched
all 32 frozen IDs on each rank: slowest prefill **5305.818 ms**, post-anchor
decode **698.474 ms** (**44.38 decode tok/s**, 31 timed tokens; **5.33 output
tok/s** over the combined engine prefill/decode spans). In the MTP-on arm,
all four ranks returned exit 1 at load: `GRIMOIRE_QFN_TP4 needs four TP
ranks, dense safetensors, --proj bf16, and speculation off`. Receipt:
`stage=mtp-on exit=1`; no MTP-on tokens or verifier rounds, so neither
accepted/step nor throughput exists for that arm. `serving-after.log`
verifies the frozen serving hash.

Do not remove the serving refusal. Code recheck: `tp_enabled()` excludes
Megatron TP, so the generic non-batch TP guard is not the blocker for this
TP4 plan. The Qwen4-Exp verifier requires `mtp4.ok` and rollback scratch,
which are absent with MTP off. The CLI-only verifier diagnostic below
allocates the scratch independently and injects deterministic proposals at
the production draft call. Live MTP remains off until a separate serving
qualification.

## E — HC projection split-K candidate

The fenced per-projection receipt (`diagnostics/act-hc-projections-KXcJl2/`,
`stage=verify exit=0`) measured these totals across all six frozen chunks:

| Rank | Down ms | Up ms |
|---|---:|---:|
| 0 | 1011.4 | 232.9 |
| 1 | 1015.4 | 233.3 |
| 2 | 1007.2 | 233.4 |
| 3 | 1009.9 | 233.1 |

Chose the down projection: vLLM patch 0003 splits its 10,240-wide K axis
into two independent 5,120-wide BF16 products, then adds their FP32 outputs
in fixed order. Reused native XMX tile code, with no precision or weight
format change. This changes FP32 accumulation order. The candidate
executed on all four ranks in every chunk, but failed the frozen gate at
**generated token 23** (`20438` control vs `9468` split-K). Receipt:
`diagnostics/act-hc-split-AoajqO/receipt`, `stage=parity exit=1`.
The isolated build succeeded (`diagnostics/act-hc-split-build-6H4qdN/build.exit=0`).
The receipt was already present on recheck: all eight rank processes exited
0; both arms ran all six chunks. The recorded launch set `QFN_HC_TEST=split`,
which put `GRIMOIRE_QFN_HC_DOWN_SPLIT_K=1` in the candidate environment,
not `GRIMOIRE_QFN_HC_FUSED_PREFILL=1`. The runner named the candidate
`fused-gate`, but each of its four logs reports `QFN HC prefill split`
with 96 calls in each chunk, and none reports nonzero fusion calls. The control's 32 IDs match the
prior grouped control on all ranks; the candidate's 32 IDs agree with each
other on all ranks, not with the control. The parity refusal correctly
prevented both trace-off timing arms. There is no paired down-projection or
end-to-end timing to recover, and no missing gate arm to rerun. The serving
SHA-256 was rechecked after the failed gate.

| Workload, MTP off | Prefill tok/s (target 4,000) | Decode tok/s (target 140) |
|---|---:|---:|
| Prior grouped TP4 control, trace off | 511.64 | 46.34 |
| HC-down split-K candidate | Not timed: parity failed | Not timed: parity failed |

Split-K was removed from the candidate runtime. This mismatch is separate
from PP4 token 19. Its replacement (`GRIMOIRE_QFN_HC_DOWN_TILED=1`) uses
64 instead of 256 output columns per workgroup in the native packed BF16
GEMM, keeping the packed pitch, subgroup fragments, full K-loop and FP32
accumulator unchanged. `regressions/2026-11-20-qfn-hc-down-tiles/` compares
every FP32 output bit at M=512 and M=146 against `launch_gemm_xmx`. The
runner requires that check, all six candidate chunks per rank, and all
32 IDs against both the same-binary control and the original grouped
receipt before timing. Build: `diagnostics/act-hc-tiles-build-TmSnVu/`
(`build.exit=0`, `job.exit=0`). Run: `diagnostics/act-hc-tiles-W9M72Y/`
(`receipt`: `stage=verify exit=0`). `hc-bitwise.log` reports all 163,840 and
46,720 FP32 outputs identical. `parity.log` reports all 32 IDs identical;
anchor logit `29.071537`, margin `12.4054375` both arms. `timing.tsv`,
one trace-off pair (runner label `fused` here means tiled):

| Arm | Prefill ms | Prefill tok/s (target 4,000) | Decode ms | Decode tok/s (target 140) |
|---|---:|---:|---:|---:|
| Grouped control | 5293.570 | 511.19 | 675.197 | 45.91 |
| HC-down output tiling | 4815.086 | 561.98 | 684.996 | 45.26 |

Prefill saved 478.5 ms (9.0% of wall time) in this pair, but is still 7.12×
the budget. Decode was unchanged in code; one pair does not establish that
the 9.8 ms drift is systematic. Keep the candidate opt-in pending a repeated,
order-reversed A/B. Serving SHA-256 matched after the run.

## E/F — Independent baseline bottleneck profile

Frozen 2,706/32, TP4 grouped prefill, native one-row expert decode, peer
push, MTP and HC candidates off. The detailed diagnostic uses
`GRIMOIRE_QFN_TP_PREFILL_REDUCE_TIMING=1`, `GRIMOIRE_Q4_TIMING=1` and
`GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1`. Build exit 0:
`diagnostics/act-qfn-phases-build-L4ijCY/build.exit`. Run:
`diagnostics/act-qfn-phases-z36CTD/receipt` (`stage=verify exit=0`),
`profile.tsv` and `breakdown.txt`. Each rank's 32 IDs match the prior grouped
receipt. The fences change execution timing; these are diagnostic spans,
not throughput measurements.

| Rank | Prefill wall (ms) | Prefill TP transport | Prefill grouped routed path¹ | Decode wall (ms) | Decode MoE stage² | Decode DeltaNet stage² | Decode TP transport |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5438.568 | 768.277 | 1628.845 | 767.862 | 341.586 | 152.614 | 185.221 |
| 1 | 5438.436 | 767.305 | 1689.763 | 767.861 | 342.130 | 153.456 | 214.001 |
| 2 | 5438.544 | 718.830 | 1740.897 | 767.859 | 342.195 | 153.440 | 217.661 |
| 3 | 5438.399 | 682.778 | 1772.057 | 767.861 | 342.232 | 153.659 | 216.637 |

¹ Route plan plus the fenced routed device path, not just the GEMMs.
² Inclusive of that stage's transport. Decode is not grouped XMX; it uses
rank-local INT4 GEMVs. Do not add the inclusive stages to the transport.

- **Prefill:** budget **676.5 ms**. On slowest end-to-end rank 0, MoE is the
  largest named stage at **2996.1 ms** inclusive, **2501.1 ms** excluding
  its transport; of that, **1628.8 ms** is the grouped routed path (the
  largest routed total is **1772.1 ms**, rank 3). DeltaNet is **720.0 ms**
  inclusive / **517.1 ms** excluding transport; HC is **1455.6 ms**. The
  582 reductions on rank 0 take **768.3 ms** excluding queue drain (1.6 ms).
  The 64→128-column grouped MoE test, below, did not help. Next: pipeline
  packed expert weights along K while preserving the dot product order.
- **Decode:** budget **229 ms** for 32 emitted IDs (the phase timer counts
  31 post-anchor steps, whose 140 tok/s budget is 221.4 ms). On slowest
  end-to-end rank 0, MoE is the largest named stage, **341.6 ms** inclusive,
  **235.1 ms** excluding its transport; **153.4 ms** is routed experts.
  DeltaNet is **152.6 ms** inclusive / **95.4 ms** excluding transport.
  Across 3007 calls on that rank, queue drain **140.8 ms** includes work
  already attributed to the surrounding stages. Transport is **185.2 ms**:
  copies **38.2**, peer readiness **80.2**, rank-order sum **27.8**, and
  peer consumption/reuse **39.1 ms**. Largest transport total is **217.7 ms**
  on rank 2, not a sum over ranks. Next decode implementation: optimize the
  rank-local INT4 expert GEMV launches, starting with avoiding the standalone
  route remap by applying the resident-window check during expert dispatch.
  Keep the selected experts and summation order unchanged and gate all 32 IDs
  before measuring. Test transport changes separately; peer waits can reflect
  other ranks' workload imbalance, not just synchronization overhead.

### Grouped MoE width screen — 64 vs 128 output columns

`GRIMOIRE_QFN_GROUPED_WIDE_N=1` reuses each staged activation tile for
more output columns without splitting K. Initial `act-grouped-wide-Y6s5py`
stopped at link (the old JIT library lacked the HC-tiled entry point);
no ranks ran. The retry linked the private rebuilt library with MTP and
HC tiling off in both arms. `act-grouped-wide-build-yu6fy5/` has
`build-retry.exit=0`, `retry.exit=0`; bounded `tmux wait-for` collected
`act-grouped-wide-retry-GZv4yv-final`. Run `act-grouped-wide-retry-GZv4yv/`
has `stage=verify exit=0`. Both arms completed all six chunks on all four
ranks and matched all 32 IDs; both report first logit `29.071537`, margin
`12.4054375`. The runner checked the frozen grouped control and serving hash.

| Trace-off arm | Prefill ms | Prefill tok/s (target 4,000) | Decode ms | Decode tok/s (target 140) |
|---|---:|---:|---:|---:|
| 64 columns | 5291.550 | 511.38 | 670.501 | 46.23 |
| 128 columns | 6140.367 | 440.69 | 711.990 | 43.54 |

Do not promote: this pair cost **848.8 ms more prefill**. The code change
is prefill-only; the decode drift is not established as causal by one pair.

### What the fast vLLM recipe actually does

- **Observed last launch**, not a new benchmark: lane 1 in
  `~/AI/b70-recipe-launcher/switch.py` uses `engines/four-cards/env/tp4.env`
  and `tp4-local.env`. `~/AI/qwen38-flash-tp2/engine-data/engine.log:1,50`
  records BF16 TP4, EP enabled, 1024 batched-token budget, 16 sequences,
  `FULL_AND_PIECEWISE` graphs, async scheduling, utilization .95, and MTP3
  with `/models/qwen38-w4a16-mtp-snapshot`. The launcher README's 4.6–5.2K
  prefill figures are historical, not a matched frozen 2706/32 receipt.
- **Routed MoE:** log lines 138–139, 200–201 report linear EP4 (128/512
  experts/rank), XPU WNA16, `NoDPEPModular`, and `XPUExpertsWNA16`.
  That is whole-expert placement with replicated token rows, not cross-rank
  token all-to-all. The hot/cold, CPU expert and custom decode-op paths are
  off in `tp4-local.env:23–28`. The sibling source
  `~/containers/b70-lane/build/rootfs/home/haxor/vllm-xpu-kernels` at
  `5f72068f1c07` shows device remap / rows-per-expert, two Xe2 grouped
  INT4 GEMMs around activation, and weighted device gather
  (`vllm_xpu_kernels/fused_moe_interface.py:353–503`). The scheduler walks
  device expert offsets; `csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:344–347`
  has six K tiles in flight for the packed-weight path. These are
  source-verified implementation details, not a trace of the image's exact
  kernel binary. In GRIMOIRE, route IDs are read back to build descriptors
  on the host, and each XMX workgroup stages packed B as BF16 into shared
  local memory with barriers on every K tile.
- **TP communication:** log lines 50, 65, 78, 88, 98 show custom all-reduce
  disabled and backend XCCL on all four ranks. `tp4.env:12–14` enables the
  SYCL all-gatherv/all-reduce temp buffers and socket IPC handle exchange.
  The vLLM sibling source `fused_moe/runner/moe_runner.py:477–509` reduces
  the dense activation (BF16 in this configuration), instead of exchanging
  routed token rows. The log does not prove whether physical payloads
  bounce through host memory. GRIMOIRE `PeerPush4` explicitly pushes each
  original **FP32** partial to the three peers and every rank sums in rank
  order, with both readiness and reuse handshakes. Do not infer the XCCL
  algorithm or copy BF16 wire rounding into the exact native gate.

**Next implementation, not another tile sweep:** pilot a prefetch pipeline
for packed INT4 expert B and its scales/zeros in the native grouped GEMM.
Start with the existing 64-column geometry and a bounded lookahead for the
next K block; keep the BF16 unpack rounding, FP32 DPAS accumulation order,
router, and transport unchanged. The measured routed device path is
1.60–1.74 s versus 31–36 ms for route planning, so attack packed loads
before moving descriptors to the device. First measure gate/up and down
separately and prove bitwise projection equality on captured resident
routes, including uneven tails and an empty rank. Then require all 32
frozen IDs and a trace-off pair. Separately prototype native FP32
reduce-scatter/all-gather with the same ascending-rank reduction order;
compare on real payload sizes and do not assume the MoE change solves the
transport budget.

### Standalone prefill lane

`../GRIMOIRE-qfn-collective-timing/diagnostics/act-standalone-prefill-49hlGe/`
is a fresh `ff98ff4` archive plus only the existing prefill timing hunks.
The first build succeeded but the patch had not landed: its run correctly
refused to produce a profile (`run/receipt`, `stage=profile exit=1`). The
retry explicitly applied the patch and verified both the env hook and log
format in the source before building (`build-retry.exit=0`). `run-retry/`
exited at `stage=lock exit=1` while the HC benchmark used the GPUs; the same
binary ran with a longer, bounded GPU-lock wait. Collected the final tmux
marker using `timeout 155s tmux wait-for act-standalone-prefill-49hlGe-qualified-done`.
`qualified.exit=0`, `run-qualified/receipt`: `stage=verify exit=0`; all four
ranks matched the frozen 32-ID control. The runner verified the serving hash.

| Rank | E2E prefill ms | Fenced layer span ms | TP reductions | TP transport ms, excluding drain | Queue drain ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 5431.512 | 5273.218 | 582 | 764.202 | 1772.158 |
| 1 | 5431.552 | 5270.072 | 582 | 762.012 | 1771.771 |
| 2 | 5431.327 | 5270.227 | 582 | 709.593 | 1771.394 |
| 3 | 5431.374 | 5270.846 | 582 | 679.604 | 1772.022 |

On slowest end-to-end rank **1**, reduction transport alone is **762.0 ms**,
**85.5 ms over the 676.5 ms entire prefill budget** (1.13×). On that rank:
PLE **65.9**, DeltaNet **195.8**, attention **64.4**, MoE **436.0 ms**. The
maximum reduction total is **764.2 ms**, rank 0, not the sum of all ranks.
Queue drain is outstanding device work and is not added to other compute
stages. The layer fence profile is diagnostic, not a speed-test result.
Prioritize grouped routed GEMM, but the transport must shrink too: even
removing every other cost in this profile would not hit 4,000 tok/s.

### Prefill chunk sweep — pinned grouped-XMX binary, no MTP build dependency

`diagnostics/act-chunk-sweep-7vMDn4/` (`job.exit=0`, `receipt`:
`stage=verify exit=0`) reused the exact binary and prompt from
`act-xmx-ab-N8CMZj/inputs.sha256`. All three sizes executed grouped XMX on
every requested chunk on all four ranks and matched all 32 frozen TP4 IDs
before timing. Timed reruns also matched all IDs. Reduction counts are
**per rank and logical**: the pinned binary emitted call and sub-chunk
shapes, not a counter, so `reductions.tsv` derives the counts from the
48 MoE sites per call and 48 mixer + 1 PLE sites per sub-chunk. The 512
result agrees with the independent measured 582-call receipt. Peer push
may split a logical reduction larger than its 8 MiB slot.

| Chunk | Actual call rows | Reductions/rank | Trace-off prefill ms | Prefill tok/s | Decode tok/s |
|---:|---|---:|---:|---:|---:|
| 512 | 5 × 512 + 146 | 582 | 5274.435 | 513.04 | 46.16 |
| 1024 | 1024 + 1024 + 658 | 291 | 3869.677 | 699.28 | 45.69 |
| 2048 | 2048 + 658 | 194 | 3657.111 | 739.93 | 46.21 |

2048 was fastest in this sweep: 30.7% lower prefill latency than 512.
This is one pass over one prompt; neither target is met and no serving
default changed. The bounded tmux collection and `serving-after.log`
confirm completion and the unchanged PP4 serving SHA.

### Device-side grouped MoE route packing

`GRIMOIRE_QFN_DEVICE_ROUTES=1` builds expert counts, stable offsets,
permutation, inverse, and tile descriptors on the rank's GPU, with no route
readback. The capacity-sized launches guard against the actual device row
and tile counts. `act-device-routes-build-OZKZ7Z/` built successfully;
`act-device-routes-3vTyX3/` has `stage=verify exit=0`. The native fixture
matched 36 rank-local plans against the host oracle, all four ranks matched
all 32 frozen IDs, and both arms had first logit 29.071537 and margin
12.4054375. The trace-off same-binary pair measured **511.56 → 515.60 prefill
tok/s** (5289.658 → 5248.266 ms) and **45.50 → 46.31 decode tok/s**
(681.379 → 669.433 ms). Only prefill code changed; do not credit the decode
movement to the packer. One pair is not enough to promote it. Serving SHA
verified unchanged.

### Native TP4 MTP-off decode projection profile

The isolated `src/grimoire.cpp` now brackets resident routed INT4 gate/up
and down separately, and HC down/inject and up/mean separately. The existing
peer-push reduction buckets record logical calls and payload size (total,
min, max), separating preceding queue drain from copy, ready, ordered sum,
and reuse. This decode-only profile uses `GRIMOIRE_QFN_TP_DECODE_REDUCE_TIMING=1`
without enabling prefill timing. `act-decode-projections-MGaXLt/`
(`stage=verify exit=0`, build `act-decode-projections-build-jJD3ZX/`)
matched all 32 frozen IDs on all ranks. Each of the 31 post-anchor steps
ran 97 reductions: 1 PLE, 36 DeltaNet, 12 attention, and 48 MoE. Each rank
reported **3,007 logical reductions × 10,240 bytes** (30,791,680 bytes;
logical FP32 partials, not the sum of link traffic).

| Rank | INT4 gate/up + down ms | HC down/inject + up/mean ms | `tp_block_reduce` transport ms, excl. drain | Ready + reuse ms |
|---:|---:|---:|---:|---:|
| 0 | 109.696 + 51.675 | 75.534 + 66.171 | 184.989 | 80.163 + 40.171 |
| 1 | 113.815 + 54.005 | 75.380 + 65.744 | 181.893 | 77.131 + 37.553 |
| 2 | 112.447 + 53.315 | 74.566 + 65.256 | 185.098 | 79.042 + 40.658 |
| 3 | 112.176 + 53.759 | 75.796 + 65.687 | 183.137 | 77.776 + 39.993 |

Transport was largest on all four ranks even without the 140.8–141.6 ms
preceding queue drain. The largest transport component was readiness.
`GRIMOIRE_QFN_TP_DECODE_FANOUT=1` sends each ready/consumed packet to all peers
before waiting for replies, rather than serializing peer exchanges. It
preserves both barriers, FP32 payloads, and ascending-rank addition. It
applies only to TP4 target decode reductions. The CPU four-rank socket
regression passed 1,024 phases and rejected mismatched sequence numbers
(`act-decode-fanout-build-bG5T9A/host.log`). `act-decode-fanout-uHmbJH/`
completed with `job.exit=0`, `stage=verify exit=0`: both gate arms and both
trace-off arms matched all 32 frozen TP4 IDs on all four ranks. Fenced
transport fell from 188.1–221.7 to 148.3–152.5 ms across ranks, excluding
queue drain. The trace-off same-binary pair measured:

| Arm | Prefill ms / tok/s | Decode ms / tok/s (31 steps) |
|---|---:|---:|
| Peer-push control | 5296.891 / 510.87 | 682.357 / 45.43 |
| Decode packet fan-out | 5276.808 / 512.81 | 644.131 / 48.13 |

One pair: 38.226 ms less decode time (5.6%), still short of 140 tok/s.
Prefill does not use fan-out; do not credit the prefill movement to it.
The opt-in stays off by default. PP4 serving SHA verified unchanged.

### Diagnostic-only TP4 frozen verifier

The one-shot CLI (`GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=1`) loads native TP4
with real drafting disabled. It injects `[1156, 16562, 16563]` at the
production `generate_tokens` draft call and runs the existing batched
verifier, TP agreement, and `commit_spec_prefix`. The gate requires
proposal two to be rejected, leaving anchor 760 and proposal 1156 committed
at position 2708 and correction 16561 pending. It compares the actual
committed token buffer and cursor, every DeltaNet state and convolution-ring
element, and the rank's full next-logit shard against fresh target-only
replay at position 2708 and after consuming the correction at 2709. The
verification logits come from the accepted row, not the last rejected row.
Exact comparison is required and the receipt reports numerical differences
without treating same greedy IDs as a state pass.

The updated `bin/test_generation` regression failed to compile before the
hook existed, then passed with the injected callback and no MTP drafting
(`act-tp4-verifier-build-siIh8U/host.exit=0`); `make test-correctness`
also passed. The isolated SYCL build passed (`build.exit=0`). The live GPU
receipt `act-tp4-verifier-QsDyCz/` completed with `job.exit=1`,
`stage=tp4-verifier-compare exit=1`: all four ranks ran batched verification,
accepted exactly two rows including the anchor, and agreed on correction
16561. Committed token buffers and cursors were exact at both positions,
and the next greedy ID after correction was 864 on every rank in both paths.
However, recurrent state, convolution rings, and next logits differed on
**all four ranks**. There were no non-finite values. The largest absolute
error across ranks for each comparison was:

| Position | Recurrent state | Convolution ring | Next logits |
|---:|---:|---:|---:|
| 2708, committed prefix | 0.00327729 | 0.04966164 | 0.04982364 |
| 2709, correction replayed | 0.00296843 | 0.04966164 | 0.03253078 |

`verifier.log` records per-rank sizes, differing counts, and RMS values.
The receipt does not distinguish batched-vs-sequential arithmetic from a
rollback defect. Matching greedy IDs do not qualify live MTP. The CLI
opt-in does not change the TP4 serving loader guard or serving generation
options. PP4 serving SHA verified unchanged after the run.

#### Same-batch rollback control

`act-tp4-verifier-same-batch-uJzLqv/` adds an independent device checkpoint
at verifier row 1, immediately after token `1156`. The DeltaNet kernel saves
its live recurrent registers after that row; the convolution kernel saves
its channel history before the next row. The checkpoint does not read
`spec_dn_inputs`, `spec_conv_ring`, or `spec_conv_inputs`. The test compares
against `commit_spec_prefix` at position 2708, then consumes correction
`16561` from the restored state and from the independent checkpoint. The
second branch starts from a fresh verifier pass and replaces only the
recurrent state and convolution ring. This is not an independent check of
KV or PLE rollback. The target-only sequential comparison remains separate.

**Result:** the isolated SYCL build and `make test-correctness` both passed
(`act-tp4-verifier-same-build-13LQFC/{build,host}.exit=0`). Bounded tmux
collection returned `stage=tp4-verifier-sequential-compare exit=2`: the
same-batch control passed on **all four ranks**, while the sequential
control remained non-exact on all four. At position 2708, each rank's
7,077,888 recurrent values and 276,480 convolution-ring values matched
bit-for-bit against the independent checkpoint. The second verification
reproduced that checkpoint exactly. After correction at 2709, each rank's
62,080 logits and both state arrays also matched bit-for-bit between restore
and checkpoint; both predicted `864`. No comparison contained nonfinite
values. The separate sequential control reproduced the preceding max errors
(including logits 0.04982364 at 2708 and 0.03253078 at 2709).

For this frozen rejection, recurrent and convolution rollback preserve
exactly the state computed inside the batch. The sequential discrepancy is
already present in the state produced by the batched verifier; it is not
explained by a defect in restoring these two buffers. This does not qualify
other inputs, KV/PLE rollback, or live MTP. Serving guard unchanged; PP4
serving SHA verified after the run.

#### Native MTP3 verifier diagnostic

`GRIMOIRE_QFN_TP4_VERIFY_DIAGNOSTIC=mtp3` selects the one-shot qgate
CLI path. It keeps `GRIMOIRE_MTP=0` at load, loads the external draft
snapshot, and runs the real three-deep `mtp_draft` path through the same
ranked generation, batched verifier, and commit code. The TP4 serving
loader still requires speculation off. The diagnostic populates the
head's KV/indexer context across the entire prompt, including rows outside
its existing 256-row warm scratch tile. The fixed-proposal rollback
control remains separately available.

`QFN_BENCH_MODE=mtp3-diagnostic` checks the off/on arms against all 32
frozen TP4 IDs on each rank and requires actual verification rounds in
the on arm. The trace-off timing pairs and separate fenced phase profile
are conditional on both passes. The exact batched-vs-sequential state
and logits discrepancy remains open regardless of a token-gate pass.

**Result:** `diagnostics/act-native-mtp3-AuDuZ9/receipt` is
`stage=verify exit=0` (bounded `timeout … tmux wait-for`). Both arms
matched all 32 frozen IDs on all four ranks in the gate and trace-off
runs; the fenced profile also matched. `make test-correctness` and the
isolated native SYCL build passed (`act-native-mtp3-build-VJqdta`).

| Same-binary arm (one pair) | Prefill ms | Decode ms (31 post-anchor tokens) | Decode tok/s |
|---|---:|---:|---:|
| MTP off | 5308.956 | 683.435 | 45.36 |
| MTP3 diagnostic | 5707.298 | 1088.829 | 28.47 |

MTP3 accepted **21 of 33 proposals across 11 rounds** (63.6%,
**1.91 accepted proposals/round**, not counting the anchor), identically on
all four ranks. The separate fenced pass measured **4.28–4.29 ms draft**
and **91.76–91.99 ms verify** per round across ranks; verify includes
warming draft-cache rows from the target batch. These fenced timings are
not the throughput measurement. The single trace-off pair was **37.2%
slower in decode throughput** with MTP3, and the verifier dominates the
measured speculative round. Keep this path diagnostic-only; it does not
qualify exact sequential arithmetic or live serving. PP4 serving SHA-256
was verified unchanged after the run.

#### Level Zero IPC event synchronization prototype

Isolated qgate `GRIMOIRE_QFN_TP_SYNC=ipc-events` exports one event pool
per peer link at setup. Every transport chunk uses fresh `ready` and
`consumed` events; the pool is capped at 4,096 chunks so another rank
cannot observe a stale signal from a reused event. A native copy waits
for the receiver's preceding `consumed` before rewriting the same receive
buffer. The sum waits for all three incoming and outgoing `ready` signals,
then the sum's completion releases the receive buffers. Both transports
use the same FP32 `((rank0 + rank1) + rank2) + rank3` kernel. Sockets
remain for setup, teardown, and unrelated rank control, not the event
arm's per-chunk ready and consumed barriers.

The control (`GRIMOIRE_QFN_TP_SYNC=fanout`) uses socket fan-out for both
prefill and decode. The gate and trace-off pair use the same grouped-XMX
binary, 2048+658 prefill rows, 8 MiB transport chunks, and MTP off. The
runner compares each rank's 32 IDs against the frozen TP4 reference and
withholds timing summaries on a mismatch.

First version (`diagnostics/act-ipc-events-Q9RqGr`): the traced gates passed
on all ranks, but the trace-off IPC arm failed at generated token **13**
on all four ranks (`6558` expected, `20482` actual); receipt
`stage=1-ipc-events-parity exit=1`. The qualified control alone ran at
**730.98 prefill / 48.08 decode tok/s** (3701.888 / 644.745 ms). No qualified
IPC speed comparison. This result does not identify the source of the
mismatch. The follow-up variant joins all six ready events with a native
Level Zero barrier and passes one local join event to the SYCL sum.

Native-join follow-up (`diagnostics/act-ipc-events-join-Gegc4H`): socket
fan-out again matched 32/32 on every rank. The IPC arm exited normally
on all ranks but failed the traced gate at generated token **19** on
all four (`15060` expected, `364` actual); receipt
`stage=ipc-events-gate-parity exit=1`. The runner skipped the trace-off
pair for that version. Neither version qualifies; retain the prototype
opt-in only and keep socket synchronization. Neither failure establishes
whether a dependency or cache-visibility issue caused the mismatch.

Both isolated SYCL builds and `make test-correctness` passed. The runs
were collected with bounded `timeout … tmux wait-for`; the PP4 serving
SHA-256 was unchanged. The TP4 serving guard, diagnostic-only MTP3, and
open sequential arithmetic mismatch are unchanged.

#### Grouped MoE projection profile and K prefetch

`diagnostics/act-moe-projections-geolD4` (`stage=verify exit=0`) used
socket fan-out, grouped XMX N64, and 2048+658 prompt rows. The SYCL profiling
queue measured only each grouped GEMM's device start to end; it did not
include routing, SwiGLU, unpermute, or reductions. Each projection ran 48
kernels per chunk and all four ranks matched the 32 frozen IDs.

| Rows | Gate/up total per rank (ms, min–max) | Down total per rank (ms, min–max) |
|---|---:|---:|
| 2048 | 412.245–460.104 | 194.303–216.700 |
| 658 | 200.668–228.324 | 93.188–104.790 |

Gate/up was the slower projection on all ranks, so only it gets the
opt-in `GRIMOIRE_QFN_GROUPED_K_PREFETCH=1` path in `src/gemm_xmx.cpp`.
The fast vLLM lane's Xe2 grouped kernel (`gemm_xe2.hpp`, six-tile
`prefetch_dist`) primes a lookahead window and refills it while consuming
K tiles. This adaptation applies that schedule to the existing K32
kernel using subgroup 2D prefetch hints: bounded full activation tiles
and aligned 64-byte packed-weight strips (one strip covers four K32
steps). Scale/zero loads, BF16 staging and the FP32 DPAS sequence are
unchanged. Output tiles stay 64 columns; the down kernel is unchanged.

`diagnostics/act-moe-kprefetch-JSgWhi/receipt` is `stage=verify exit=0`.
The same-binary gate confirmed 48 prefetched gate/up launches in each
chunk and matched all 32 frozen IDs on all four ranks in both arms.
The trace-off pair rechecked that match. The isolated SYCL build and
`make test-correctness` passed (`act-moe-kprefetch-build-SJOx57`).

| Trace-off arm, socket fan-out | Prefill ms | Prefill tok/s | Decode ms (31 post-anchor tokens) | Decode tok/s |
|---|---:|---:|---:|---:|
| Grouped N64 control | 3697.348 | 731.88 | 660.658 | 46.92 |
| Gate/up K prefetch | 3729.092 | 725.65 | 634.351 | 48.87 |

The candidate added **31.744 ms (0.86%)** to prefill in this pair. Do
not credit the decode movement to a prefill-only kernel change. There
is no measured prefill win; keep it off by default. Both receipts were
collected with bounded `timeout … tmux wait-for`; the PP4 serving SHA-256
was unchanged. Socket fan-out remains the control, IPC and MTP3 remain
diagnostic-only, and the TP4 serving guard is unchanged.

#### Device-offset scheduler for grouped gate/up

`GRIMOIRE_QFN_GROUPED_OFFSET_SCHED=1` replaces only the N64 gate/up
capacity grid in `src/gemm_xmx.cpp`. Inspired by vLLM's
`PersistentTileSchedulerMoE` (`moe_tile_scheduler.hpp`), it launches one
workgroup per reported compute unit. Worker `g` walks linear tiles
`g, g + pool, …`; its cursor reads the live expert prefix offsets
produced by `src/qfn_device_routes.hpp`, skips empty experts, and stops
at the last active tile. No global work counter, route readback, or
K splitting. The capacity-grid control and candidate use the identical
M32/N64 tile body and increasing-K FP32 accumulation. Down still
consumes the same device-built descriptors in both arms, so route
packing itself is unchanged.

The native host regression checks exact work coverage against the
existing descriptor builder for empty and sparse experts, 31/32/33-row
edges, and multiple workgroup strides. `make test-correctness` and the
isolated SYCL build passed (`act-moe-offset-build-cQW5t9`). The paired
gate uses device routes and socket fan-out with 2048+658 rows in both
arms; K prefetch, IPC events, and MTP are off.

`diagnostics/act-moe-offset-EeOag2/receipt` is `stage=verify exit=0`.
Both gate arms and both trace-off timing arms matched all 32 frozen IDs
on all four ranks. The candidate dispatched **256 workgroups** per
launch on each rank, with 48 gate/up calls in each prompt chunk.

| Same-binary trace-off arm | Prefill ms | Prefill tok/s | Decode ms (31 post-anchor tokens) | Decode tok/s |
|---|---:|---:|---:|---:|
| Device routes, capacity grid | 3665.994 | 738.14 | 668.379 | 46.38 |
| Device routes, offset scheduler | 4194.395 | 645.15 | 650.391 | 47.66 |

The fixed pool added **528.401 ms (+14.4%)** to prefill in this pair.
Keep the capacity grid; the opt-in prototype is not a performance win.
Decode movement is not credited to this prefill-only change. Collected
with bounded `timeout … tmux wait-for`; the PP4 serving SHA-256 was
unchanged and the TP4 serving guard was not modified.

#### Full vLLM Xe2 W4A16 gate/up comparison

`regressions/2026-11-20-qfn-xe2-gate-up/run.sh` builds a native SYCL
observer from a temporary copy of qgate. It submits the unmodified
`MoE::MoEGEMM` from the lane's `vllm-xpu-kernels` `5f72068f1c07` and
SYCL-TLA `87f6850680a5`, not a rewrite of the scheduler or mainloop.
It uses the lane's M128/N256/K32 policy (`20480 / 128 = 160` allocated
rows per expert) and the same BF16 permuted rows, rank-local routes,
INT4 codes and scales as qgate's N64 capacity grid. Samples are from
layers 0, 24 and 47 of the first 2,048 frozen prompt rows. The N64 control
still feeds the model, with device routes and socket fan-out. The
Xe2 timing arm charges the atomic counter clear and BF16-to-FP32 widening;
it would also report the full XOR cost for unsigned zero-8 inputs.

The first harness (`act-xe2-uIpUqM`) incorrectly refused GRIMOIRE's
`0xff` signed-layout marker as a non-8 zero point. The CPU-only stored-byte
audit (`act-xe2-EGJkxi/audit.log`) found literal runtime zero 8 after
decoding the GPTQ zero-minus-one field for all 39,321,600 sampled groups
and centered codes in `[-8,7]`. The model loader had already applied the
lossless fold at load time. Both kernels used the same resident s4 bytes
in the corrected observer; no per-call weight conversion was needed.

`act-xe2-EGJkxi` built cleanly and reached the layer-0 projection on
ranks 1 and 2. Against the FP32 control, **2,651,650 / 5,573,120** and
**3,962,029 / 8,409,600** values, respectively, were outside the BF16
rounding envelope (relative RMS **0.275% / 0.274%**). This fails the
predeclared projection gate. It does not isolate the discrepant
arithmetic to the epilogue or in-register scaling. The driver also
reported CCS memory faults and timed-out jobs on ranks 0 and 3 before
they emitted a projection (`xe-faults.log`); ranks 1 and 2 later
reported a failed mixer all-reduce. The run exited 124 with TERM-only
timeouts. The PP4 serving SHA-256 matched after the run.

The retry `act-xe2-ZSglw0` matches the lane's output allocation
capacity and launches widening with a device live-row predicate. The
native build and host checks passed; all ranks timed out in load/setup
before prefill (`job.exit=124`). The retry did not test whether the
capacity change fixes the earlier faults. No render-node users remained
after exit. No power cycle or serving changes; the serving SHA-256
matched again. **No qualified projection timing, candidate wiring, or
frozen-ID timing gate.** Keep N64 capacity grid and socket fan-out.

**CPU layout regression and one-GPU follow-up.** `test_layout.cpp`
compares the two paths' packed byte/nibble order and `[expert,N,K/128]`
scale indices at the full layer-0 shape, including empty experts,
ragged M32/M128 tiles, expert prefix offsets, BF16/FP32 output strides,
and short-buffer rejection for every backing allocation. No packed
code, scale, route or stride mismatch was found. The unmodified lane's
two optional scale Block2D prefetches did violate the pinned CuTe
surface contract: the width is 2 bytes (must be a multiple of 4) and
the base advances by 2 bytes per scale group (base must be 64-byte
aligned). The isolated source overlay removes those two hints only.
There is no repacking cost on the resident s4 path. This finding does
not establish what caused the prior TP4 GPU faults.

`act-xe2-single-dl94iJ`: native build and CPU checks passed. On one
GPU, real layer-0 rank-1 weights with 4,354 deterministic synthetic
BF16 routed rows completed the projection without nonfinite outputs.
**2,786,063 / 5,573,120** values were outside tolerance (**50.0%**,
relative RMS **0.293%**, maximum absolute error **0.0092864**).
Round-to-nearest and truncation matched exactly at 2,715,640 and
2,457,572 elements, respectively, so replacing the output rounding
expectation with simple truncation does not explain the discrepancy.
The projection exited 2; timing did not run. The TP4 runner now
requires a passing single-GPU receipt and reuses its header overlay.
No TP4 retry; serving SHA-256 unchanged.

**One-hot diagnosis and BF16 rounding fix.** The layer-0-shaped fixture
probes 360 one-hot rows across three experts, on both sides of each
K128 boundary, with known signed codes and both exact and inexact
BF16 scale products. The CPU scalar oracle rounds the product to BF16
round-to-nearest-even. In `act-xe2-scale-probe-jKNNGB`, N64 matched all
460,800 values; Xe2 missed 18,991 gate and 19,007 up values, but matched
truncation on every value. A direct `apply_scale` device probe, before
DPAS or output conversion, reproduced both first wrong results:

| Projection | Code | BF16 scale bits | Oracle | Original Xe2 |
|---|---:|---|---|---|
| Gate, col 25 | +5 | `0x3f23` | `0x404c` | `0x404b` |
| Up, col 13 | -7 | `0x3f0d` | `0xc077` | `0xc076` |

`bf16-scale-rne.patch` changes only the BF16 scale multiply in the
isolated lane overlay: multiply in FP32 and explicitly convert to BF16.
No layout, FP16, DPAS, epilogue or stored code change. The fixed run
`act-xe2-rne-9ykKHI` passed the direct probe and every one-hot output,
then the real layer-0 rank-1 weight projection: **5,573,120/5,573,120**
outputs exactly match N64 rounded to BF16 (4,354 synthetic routed rows;
no nonfinite values). Eight alternating paired samples averaged
**8.287 ms N64 vs 1.322 ms Xe2** fenced wall time, including Xe2's
counter clear and **0.112 ms** device widening. No per-call weight
repack. This is a single-projection microbenchmark, not a model speed
or quality result. The real-weight single-rank prerequisite is met;
TP4 was not run, and the frozen 32-ID gate remains required before
model timing. Bounded TERM-only GPU runs; serving SHA-256 unchanged.
The prior TP4 GPU faults are not explained by this arithmetic fix.

**Opt-in full-chunk TP4 gate/up.** `run-candidate.sh` builds a qgate-only
translation unit with `GRIMOIRE_QFN_XE2_GATE_UP=1`, reusing the exact
header overlay from `act-xe2-rne-9ykKHI`. The candidate is restricted to
full 2,048-row gate/up with signed resident weights and device routes.
It shares the control's BF16 activation rows and live device route
counts, clears the Xe2 counter, and widens the live output to the
existing FP32 SwiGLU input. The 658-row tail and all down projections
stay N64; decode is unchanged. Both arms use the same binary and socket
fan-out, with K prefetch, offset scheduling, IPC events and MTP off.

The runner aborts at the first Xe kernel fault, failed rank, or frozen-ID
mismatch. Only two passing four-rank gates admit two order-reversed
trace-off pairs. Build and `make test-correctness` passed in
`diagnostics/act-xe2-tp4-AMEqPc`.

**Frozen TP4 result: rejected.** Same-binary N64 matched all 32 IDs on
all four ranks. Xe2 logged 48 candidate calls per rank for the 2,048-row
chunk and completed the N64 tail. It diverged at **generated token 19**
(1-based), **15060 → 364**. The runner stopped at the first checked
mismatch (`stage=xe2-gate-parity-rank-1 exit=1`); the already-written
logs contain the same candidate ID sequence on all four ranks. The
watcher raised no driver fault in this run. The shared first-difference
position with the old PP4/TP4 comparison is not a PP4 parity result
or evidence of the same cause. No trace-off timing arms ran. Keep
N64 capacity grid and socket fan-out as control, and Xe2 opt-in only
in the isolated diagnostic build. Final render-node occupancy was
clear; the PP4 serving SHA-256 was unchanged.

**N64 output-rounding control.** `QFN_GATE_ONLY=round` runs N64 twice;
the second arm rounds only its live gate/up FP32 output to BF16 RNE,
then widens it back in place before the unchanged SwiGLU. Both arms
use device routes, N64 capacity grid, 2,048 + 658 rows, socket fan-out,
and K prefetch off. In `diagnostics/act-xe2-round-OUSfoC`, N64 matched
32/32 IDs on all ranks. Rounding ran on 48 projections per rank and
changed ID 19 from **15060 to 364**. All four rounded-N64 sequences
matched the earlier Xe2 BF16 sequence exactly, including positions
20–32. Thus output rounding alone is sufficient to reproduce that
failure; it does not prove every intermediate is identical. The
runner stopped at the first checked mismatch
(`stage=n64-bf16-gate-parity-rank-3 exit=1`), with no timing arms.
Serving SHA-256 matched.

**Xe2 FP32 output prototype: frozen gate passed.** The existing lane
kernel now has a diagnostic instantiation with a float destination.
CuTe selects the 32-bit store; Xe2 writes the FP32 accumulator directly
into the existing SwiGLU input, with no BF16 output scratch or widening.
The lane headers are byte-identical to the qualified BF16 overlay; the
BF16 dequantization RNE fix, tile policy, routes, and weight bytes are
unchanged. In `diagnostics/act-xe2-fp32-single-zBs8Lv`, both one-hot
projections passed (460,800 total outputs) and the real-weight layer-0
rank-1 projection matched N64 FP32 bit-for-bit at all **5,573,120**
values (`outside=0 nonfinite=0`), using the same synthetic routed rows.

`diagnostics/act-xe2-fp32-tp4-PPAlLN` (`stage=verify exit=0`) ran the
same-binary N64 and Xe2 FP32 frozen gates: **32/32 IDs on all four
ranks in both arms**, with token 19 restored to `15060`. Each rank
logged 48 FP32 Xe2 gate/up calls for the full chunk; the N64 tail and
down projections were unchanged. `make test-correctness` passed.
No trace-off model timing ran; this qualifies the frozen prompt only,
not promotion or PP4 parity. The PP4 serving binary SHA-256 matched
again after the runs.

**Xe2 FP32 trace-off A/B: prefill improved in both orders.**
`diagnostics/act-xe2-fp32-ab-y5ZEfo` (`stage=verify exit=0`) reused the
exact executable from `act-xe2-fp32-tp4-PPAlLN` in all four arms, with
2,048-row chunks, device routes, socket fan-out, and K prefetch off.
Every timed arm matched all 32 frozen IDs on all four ranks. Rates use
2,706 prompt tokens and 31 post-anchor decode steps, using the slowest
rank's elapsed time for each phase.

| Pair / run order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: N64 | 3,637.588 | 743.90 | 642.005 | 48.29 |
| 1: Xe2 FP32 | 3,189.346 | 848.45 | 638.916 | 48.52 |
| 2: Xe2 FP32 | 3,207.980 | 843.52 | 634.102 | 48.89 |
| 2: N64 | 3,627.096 | 746.05 | 632.119 | 49.04 |

Xe2 saved **448.242 ms (12.3%)** and **419.116 ms (11.6%)** in prefill.
Decode moved in opposite directions across the two pairs; this is not
an observed decode gain. The Xe2 two-run means were **846.0 prefill**
and **48.7 decode tok/s**: still **3,154.0** and **91.3 tok/s** below
the 4,000 and 140 targets, respectively. Meeting them requires about
**4.73×** the measured prefill rate and **2.87×** the decode rate.
The result qualifies a repeatable prefill gain on this frozen prompt;
the path remains opt-in in qgate. Each arm's `serving.sha256` records
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

#### Native oneCCL vs PeerPush4

The fast four-card **recipe launcher** selects `b70-flash-next:tp2`
(the image tag does not describe the rank count). The installed image
`sha256:bcc91a8966e805a632f575b0467d4b8f926e1f6d7ec36cf4ecc16f3627537a05`
contains oneCCL **2022.0.0**: `readelf` on its `libtorch_xpu.so` shows a
`NEEDED` for `libccl.so.1`. That native library has SHA-256
`733980ab6a6eb15d2d3da0649b92052c64a9597ced48fe9188434face5298b35`.
`tp4.env` declares socket IPC and temporary SYCL collective buffers.
This is not the separate staged `containers/b70-lane` lab stack:
that one has oneCCL **2022.1.1** and `libtorch_xpu.so` links the
`.so.2` dispatcher with `libccl_legacy.so` bridging to `.so.1`.

`GRIMOIRE-qfn-collective-timing/regressions/2026-11-20-qfn-native-collective/`
links the installed image's `.so.1` directly through its C++ USM API.
The diagnostic uses oneCCL **allgather only** for payload transport,
then sums locally in FP32 as `((r0+r1)+r2)+r3`. A plain collective SUM
is not qualified to preserve that association. No Torch/Python linked
or loaded. It uses all four devices visible, socket IPC, the recipe's
collective temporary buffers, and single-host OFI with SYCL kernels
and SIMPLE thresholds set explicitly. The isolated extraction required
both `libfabric.so` and its provider modules; the earlier receipts
`act-ccl-image-EbXTyW` and `act-ccl-image-jzYcSG` stopped at transport
init without enough of that stack, not at GPU arithmetic. The separate
2022.1.1 narrow-affinity/pidfd lab test (`act-ccl-aH2mMw`) also passed
but is not the recipe-image benchmark below.

`GRIMOIRE-qfn-collective-timing/diagnostics/act-ccl-image-RW1DLb`
(`stage=verify exit=0`): two order-reversed pairs with all outputs
bitwise equal to an independent host rank-order oracle, including
non-associative cancellation inputs. The timer includes host completion
waits and the local GPU FP32 sum; the comparator is the actual qgate
`PeerPush4` socket fan-out with 8 MiB slots. Reported times are µs,
quantiles of the slowest rank per sample (256 decode / 48 prefill per arm).

| Run order | 10,240 B p50 / p95 µs | 20,971,520 B p50 / p95 µs |
| --- | ---: | ---: |
| 1: socket fan-out | 44.196 / 56.298 | 4,111.453 / 4,131.110 |
| 1: oneCCL ordered | 28.799 / 39.660 | 3,472.552 / 3,487.640 |
| 2: oneCCL ordered | 28.170 / 32.794 | 3,472.195 / 3,486.667 |
| 2: socket fan-out | 30.581 / 37.000 | 4,112.805 / 4,141.515 |

OneCCL ordered lowered median decode collective latency by **7.9–34.8%**
and median full-prefill collective latency by **15.5–15.6%**. These are
standalone collective latencies, not an end-to-end tok/s claim.

**Xe2 FP32 qgate qualification: passed.** `model-gate.sh` builds one
isolated diagnostic executable from the saved Xe2 FP32 translation unit
and objects. `GRIMOIRE_QFN_TP_SYNC=ccl-ordered` substitutes ordered
oneCCL only for the two measured sizes; other sizes (including the
658-row tail) still use the socket fan-out control. It does not enable
an unordered SUM or fall back on a oneCCL error. In
`GRIMOIRE-qfn-collective-timing/diagnostics/act-ccl-model-iLX8Co`
(`stage=verify exit=0`), both arms used Xe2 FP32 and matched **32/32
frozen IDs on all four ranks**. Each candidate rank logged **3,007
decode** and **97 full-prefill** oneCCL calls. This is opt-in qgate only.
Normal qgate and serving builds do not link oneCCL.

**Same-binary trace-off model A/B: keep socket fan-out.**
`GRIMOIRE-qfn-collective-timing/diagnostics/act-ccl-e2e-uHQyz4`
(`stage=verify exit=0`) reused the qualified executable from
`act-ccl-model-iLX8Co`. Both arms used Xe2 FP32, 2,048 + 658 prompt rows,
device routes, and K prefetch off. All four arms matched 32/32 frozen
IDs on every rank. Rates are 2,706 prefill tokens and 31 post-anchor
steps over the slowest rank's elapsed time for each phase.

| Run order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: socket fan-out | 3,255.610 | 831.18 | 660.905 | 46.91 |
| 1: ordered oneCCL | 3,267.001 | 828.28 | 680.175 | 45.58 |
| 2: ordered oneCCL | 3,153.778 | 858.02 | 649.500 | 47.73 |
| 2: socket fan-out | 3,183.905 | 849.90 | 622.697 | 49.78 |

OneCCL prefill was 11.391 ms slower in pair 1 and 30.127 ms faster
in pair 2. Decode was 19.270 and 26.803 ms slower in both pairs.
Fan-out had the lower mean total (3,861.559 vs 3,875.227 ms), so keep
it as the baseline. The collective microbenchmark gain did not give
a repeatable model-level gain. Two-run means and rate shortfalls:

| Backend | Mean prefill tok/s | Gap to 4,000 | Mean decode tok/s | Gap to 140 |
| --- | ---: | ---: | ---: | ---: |
| Socket fan-out | 840.5 | 3,159.5 | 48.3 | 91.7 |
| Ordered oneCCL | 843.2 | 3,156.9 | 46.7 | 93.3 |

**Separate fenced fan-out profile, same binary.**
`GRIMOIRE-qfn-collective-timing/diagnostics/act-ccl-profile-VFhtyk`
(`stage=verify exit=0`) also matched all 32 IDs on all four ranks.
`profile.tsv` reports phase and transport times with extra queue waits;
these times are diagnostic, not throughput:

- **Prefill, slowest rank 1** (3,252.214 ms): MoE is the largest compute
  cost at **891.462 ms** excluding its reduction (1,191.900 ms inclusive;
  local routed-expert device work alone is 720.424 ms). HC costs
  **746.500 ms**, including 418.300 ms for HC down projections. Total
  reduction transport is **573.139 ms** over 194 calls. DeltaNet is
  **512.922 ms** excluding its reduction.
- **Decode, slowest rank 0** (757.223 ms): MoE costs **245.876 ms**
  excluding its reduction (332.161 ms inclusive), of which 162.029 ms is
  routed INT4 gate/up and down. HC costs **178.500 ms**. The 3,007
  reductions cost **144.906 ms** excluding pre-reduction queue drain,
  with peer readiness the largest transport component (62.049 ms).
  DeltaNet costs **95.681 ms** excluding its reduction.

Each timed arm and the profile recorded the unchanged serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

#### Whole-expert placement and full routed-output replay

**The Xe2 FP32 socket-fan-out baseline already has the requested placement.**
`include/b70/qfn_shard_plan.hpp` assigns `rank * 128` through
`(rank + 1) * 128` (exclusive). `src/grimoire.cpp` loads complete
640-wide gate/up/down projections for each local expert; it does not
slice their intermediate width. The dense attention/DeltaNet sharding
stays TP4 (the two KV heads are shared across pairs of ranks). Like the
fast vLLM lane's linear EP4, token rows are replicated and routes are
filtered to the local expert window, not dispatched by token all-to-all.
This verifies matching placement, not matching collective or arithmetic
against vLLM.

`regressions/2026-11-20-qfn-xe2-gate-up/routed-replay-overlay.awk`
adds a diagnostic-only check to the isolated translation unit. On the
loaded model it validates all 48 local weight geometries and device
allocations per rank, plus the dense TP4 plan and KV layout. On the
first 2,048 real prompt rows, it replays N64 on the exact Xe2 BF16
input rows, weight bytes, expert descriptors, inverse route mapping
and coefficients. It compares after FP32 gate/up, BF16 SwiGLU, the
unchanged down projection and weighted unpermute, **before** the shared
expert and cross-rank reduction. The model consumes the Xe2 FP32 output;
the 658-row tail and decode retain their baseline kernels.

`diagnostics/act-xe2-routed-k7UaWd` passed `make test-correctness` and
ended `stage=verify exit=0`: all **1,006,632,960** rank-local FP32 output
values (2,048 × 2,560 × 48 × 4) were finite and bitwise equal to N64.
Each rank logged 48 Xe2 FP32 calls and matched all 32 frozen IDs after
its routed replay. Artifacts include per-rank replay logs and serving
SHA-256 receipts; the serving binary retained `61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.
No new A/B was run: outside the fenced replay, the requested placement
and kernel are the existing baseline, so there was no independent
candidate. Use the socket-fan-out rows above (mean **840.5 prefill / 48.3
decode tok/s**) as the measured reference, not the replay run.

#### Xe2 FP32 routed-expert down: exact replay and faster prefill

`regressions/2026-11-20-qfn-xe2-gate-up/` adds
`qfn_xe2::launch_signed_down` and `GRIMOIRE_QFN_XE2_DOWN=1` in the
isolated overlay. It instantiates the same pinned Xe2 M128/N256/K32
kernel and FP32 store for `K=640, N=2560`, with five scale groups per
output row and the loader's resident signed INT4 weights. No weight
repacking or new lane patch. Xe2 reads the existing packed BF16
SwiGLU output and writes directly to the FP32 down buffer; the
existing weighted unpermute is unchanged. Only the 2,048-row
prompt chunk changes. The 658-row tail and decode retain their
original kernels. Both arms keep Xe2 FP32 gate/up, device routes,
socket fan-out, and K-prefetch off.

`diagnostics/act-xe2-down-WH5q2J` passed `make test-correctness` and
finished `stage=verify exit=0`. Before timing, both down kernels
read the identical real SwiGLU input and used the same route
mapping and coefficients. The replay poisoned both live down
buffers before submission, then checked all 48 layers on each rank:
**2,516,582,400 live raw down values** and **1,006,632,960 weighted
rank-local values** were finite and bitwise equal to N64. Each rank
issued 48 Xe2 down calls and matched all 32 frozen IDs. The four
trace-off arms then reused the same binary and each independently
matched all 32 IDs on every rank. Rates use 2,706 prompt tokens and
31 post-anchor decode steps over the slowest rank in each phase.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: N64 down | 3,190.727 | 848.08 | 638.478 | 48.55 |
| 1: Xe2 FP32 down | 2,997.438 | 902.77 | 638.045 | 48.59 |
| 2: Xe2 FP32 down | 2,976.598 | 909.09 | 626.057 | 49.52 |
| 2: N64 down | 3,166.253 | 854.64 | 629.564 | 49.24 |

Xe2 saved **193.289 ms (6.06%)** and **189.655 ms (5.99%)** of
prefill. Do not credit the small decode drift to this prefill-only
change. Candidate means were **905.93 prefill / 49.06 decode tok/s**
(control **851.36 / 48.90**), leaving gaps of **3,094.07** and
**90.95 tok/s** to the 4,000 and 140 targets. Qualified for this
frozen qgate run only; keep opt-in. The replay and each timed arm
recorded and verified the frozen serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

#### Single-stream decode: the fast lane's real batch-1 dispatch

**Source-verified, in the recipe image and its pinned host source.** Lane 1
(`switch.py:45-70`, container `b70-tp2`) runs vLLM `XpuFusedMoe` at
`.../vllm_xpu_kernels/fused_moe_interface.py`. `_apply_kernel` remaps the
rank-local rows and per-expert counts, then calls
`cutlass_grouped_gemm_interface` twice: gate/up to `[rows, 2*I]`, then
down to `[rows, H]`, with `silu_and_mul` between them and `moe_gather`
after. `grouped_gemm_interface.cpp:27` routes Xe2 to
`cutlass_grouped_gemm_xe2`, and `grouped_gemm_xe2_interface.hpp:239`
computes `A_avg_M = A_total_M / num_experts`; line 357 picks
`w4a16_policy_m_8` when `A_avg_M <= 4`, i.e. **WGTile M8 / N64 / K32**
for both projections. With MTP3 a decode step carries 1 + 3 rows, so
`A_total_M = 40` over 128 local experts and `A_avg_M = 0` — the same
M8 policy. **There is no GEMV and no M=1 backend switch** in this path.
The recipe's separate fused decode op is off (`tp4-local.env:27`
`B70_MOE_DECODE_OP=0`), and lane 1's `engine.log` has no
`B70 decode op:` line, so that 16-row kernel is not in play.

**What 119.7 tok/s actually is** (`engine-data/bench.log`, 2026-10-20):
the mean of four single-stream requests, `106.6 / 150.1 / 96.3 / 125.6
tok/s`, at **acceptance 2.51/3.54/2.26/2.88 = 2.80 tokens per step** and
`23.4 ms/step`, so **yes, it uses MTP3**; a repeat run measured 115.25
tok/s with identical per-prompt acceptance. It is a client-side
`bench_det.py` figure, not an engine gauge.

#### Port: Xe2 W4A16 M8 routed decode (opt-in)

`qfn_xe2::DecodeM8` in
`regressions/2026-11-20-qfn-xe2-gate-up/{bench.cpp,bench.hpp}` plus
`decode-overlay.awk` and `GRIMOIRE_QFN_XE2_DECODE_M8`. On the
`forward_qwen4_exp` path only, with native TP4 peer transport, MTP and
PP off, it packs the rank-local routes into BF16 rows with per-expert
counts and slot order, runs the same pinned M8/N64/K32 kernel for
gate/up (FP32 output), casts SwiGLU to BF16 for down, runs M8 down to
FP32, and gathers weighted output in original slot order. Weights stay
the loader's resident signed INT4; no repack. Prefill stays Xe2 FP32
gate/up + Xe2 FP32 down, and transport stays socket fan-out.

**Replay against the current kernel** (`GRIMOIRE_QFN_XE2_DECODE_M8=replay`,
`diagnostics/act-xe2-decode-m8-rAfhxV`, `stage=verify exit=0`): both arms
consumed the identical real FP32 layer input and the native arm kept its
output. Every live slot's BF16 SwiGLU output and the full weighted
output were checked for **31 positions × 48 layers × 4 ranks (5,952
checks, 1,488 lines per rank)**, up to 9 local routes per rank. Worst
observed `hidden_rms=0.00422`, `weighted_rms=0.00605`, all finite, inside
the ≤1.5 % / ≤4 %-of-peak guard. **This is a numerical guard, not
bitwise equality** — admission rests on the separate candidate run
below. Candidate arm (`..._DECODE_M8=1`): 31 markers per rank with
`layers=48`, and **all 32 frozen IDs on every rank**.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: native decode MoE | 3,059.653 | 884.41 | 670.285 | 46.25 |
| 1: Xe2 M8 decode | 3,009.268 | 899.22 | 586.452 | 52.86 |
| 2: Xe2 M8 decode | 2,991.552 | 904.55 | 577.979 | 53.64 |
| 2: native decode MoE | 2,978.247 | 908.59 | 630.850 | 49.14 |

Decode improved by **83.833 ms (12.51 %)** and **52.871 ms (8.38 %)** in
the two orders; both arms run identical prefill code, so the ±1.7 %
prefill spread is run noise and is not credited. Two-run means: **Xe2 M8
53.25 decode / 901.88 prefill tok/s** vs native **47.70 / 896.50**.
Remaining gaps: **3,098.11 prefill** and **86.75 decode tok/s**
(native: 3,103.50 / 92.31). Keep opt-in and repeat before promotion; one
paired A/B is not enough. All six arms recorded and verified the frozen
serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

#### oneCCL end to end on the current stack: no win, keep socket fan-out

`act-ccl-e2e-uHQyz4` did complete (`stage=verify exit=0`, four timing
rows), but it reused the `act-ccl-model-iLX8Co` binary built on
`act-xe2-fp32-tp4-PPAlLN`: Xe2 FP32 gate/up with **N64 down and no
DecodeM8**. So transport had no number on the stack that now qualifies
prefill. `model-gate.sh` gained a `stack` mode that overlays the oneCCL
transport on any passing qgate diagnostics translation unit; it
re-verifies that unit's receipt, `inputs.sha256`, its traced arm's Xe2
gate/up **and** down markers, and its frozen IDs before composing, then
times fan-out → oneCCL → oneCCL → fan-out from one binary.

Base `act-xe2-down-WH5q2J` (Xe2 FP32 gate/up and down), chunk 2048, K
prefetch off, MTP and prefix cache off, device routes on.
`diagnostics/act-ccl-stack-6Bc0jb`, `stage=verify exit=0`: every arm
matched **32/32 frozen IDs on all four ranks**; each oneCCL arm logged
`QFN oneCCL calls: rank=r decode=3007 prefill=97` identically on all
four ranks, and the fan-out arms logged no oneCCL line.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: socket fan-out | 2,992.780 | 904.18 | 645.578 | 48.02 |
| 1: ordered oneCCL | 2,970.185 | 911.05 | 650.159 | 47.68 |
| 2: ordered oneCCL | 2,995.118 | 903.47 | 674.799 | 45.94 |
| 2: socket fan-out | 3,028.058 | 893.64 | 641.785 | 48.30 |

Decode: oneCCL **+4.581 ms** and **+33.014 ms** slower, in that order.
Prefill: −22.595 and −32.940 ms. Means are **fan-out 898.91 prefill /
48.16 decode** and **oneCCL 907.26 / 46.81**; total wall 3,654.00 vs
3,644.86 ms, i.e. **9.1 ms (0.25 %) apart** — smaller than the spread
between these arms (prefill 2,992–3,028 ms, decode 642–675 ms).

**Decision: no promotion and no DecodeM8 combination.** The stated
condition was a oneCCL win larger than noise; it loses decode in both
orders and its prefill edge is inside the noise band. Remaining gaps:
fan-out **3,101.09 prefill / 91.84 decode tok/s**, oneCCL **3,092.74 /
93.19**. Serving SHA-256 was verified unchanged after every arm. The
standalone collective benchmark still shows oneCCL 7.9–34.8 % lower
reduction latency, while end-to-end decode is slower both times; the
reason is **not established** here. A plausible but unmeasured
explanation is that a blocking gather on the in-order queue replaces
socket exchange that overlaps with compute.

#### Fresh per-phase profile on the current stack

`diagnostics/act-xe2-profile-FFTpyL` (`stage=verify exit=0`) profiles two
trace-off fan-out arms from one binary, both matching **32/32 frozen IDs
on all four ranks** with the serving SHA-256 unchanged: `profile-native`
(the `act-ccl-stack-6Bc0jb` fan-out config: Xe2 gate/up + Xe2 down, M8
off) and `profile-m8` (same plus DecodeM8). This replaces the stale
`act-qfn-phases-z36CTD` split, which predated Xe2 FP32, Xe2 down and
DecodeM8. **All times below are fenced diagnostics, not throughput**;
rate targets use the timed A/B tables above.

Prefill, mean of the four ranks (native / M8 arms are the same code):

| Phase | native | M8 |
| --- | ---: | ---: |
| wall | 3,014.1 | 3,001.3 |
| MoE inclusive | 986.8 | 989.1 |
| MoE excluding its reduction | 711.0 | 712.7 |
| routed local GEMM device work | 541.4 | 543.0 |
| route plan | 6.9 | 6.9 |
| all connections (HC) | 737.2 | 732.5 |
| — of which HC down | 410.7 | 406.1 |
| reduction transport, all kinds | 543.0 | 543.4 |
| DeltaNet (incl / excl transport) | 690.7 / 495.1 | 693.1 / 499.2 |
| full attention excluding transport | 409.8 | 409.9 |
| PLE excluding transport | 35.8 | 35.9 |

Decode, mean of the four ranks:

| Phase | native | M8 |
| --- | ---: | ---: |
| wall | 761.9 | 664.9 |
| MoE inclusive | 340.8 | 251.7 |
| MoE excluding its reduction | 256.6 | 189.5 |
| routed INT4 gate/up + down | 120.8 + 53.6 | not instrumented under M8 |
| all connections (HC) | 176.8 | 175.1 |
| — HC down inject / HC up mean | 67.1 / 65.7 | 66.1 / 65.2 |
| DeltaNet (incl / excl transport) | 138.4 / 94.6 | 140.7 / 94.4 |
| full attention excluding transport | 48.2 | 47.8 |
| reduction transport | 151.3 | 125.2 |
| — copy / ready / sum / reuse | 39.1 / 68.5 / 27.5 / 16.2 | 40.7 / 40.1 / 26.9 / 17.6 |
| head | 21.0 | 21.0 |

**Correction to the previous text:** the stale profile's "MoE 2,996 ms per
rank" is gone. On the current stack prefill MoE is **989 ms inclusive /
713 ms excluding its reduction**, and decode MoE is **252 / 190 ms**.
Routed MoE is still the largest single prefill phase, and no longer
within 2.5× of the target.

**The largest actionable prefill cost is the tail chunk, not the
placement.** The same rank spends **137.4 ms** of routed device time on
the 2,048-row chunk and **378.4 ms** on the 658-row tail — 2.8× the time
for 0.32× the rows. Xe2 gate/up and down are gated to `MR == 2048`, so
the tail still runs the N64 grouped path. A tail at the full chunk's
per-route efficiency would cost ≈44 ms, i.e. ≈334 ms recoverable, more
than every other candidate on the table.

**The EP4 conditional was not met, so that A/B was not run.** Routed MoE
is 711 ms/rank, not ~2.5 s, and whole-expert EP4 with 128 complete
experts per rank is **already the baseline**: `act-xe2-routed-k7UaWd`
validated experts `[0,128)`, `[128,256)`, `[256,384)`, `[384,512)` with
full 640-wide gate/up/down per local expert on all four ranks. Comparing
it against the socket fan-out control would compare the baseline to
itself. The next candidate is therefore extending the Xe2 FP32 gate/up
and down kernels to the 658-row tail with the lane's own M32/N64/K32
policy (its `A_avg_M <= 128` branch for that shape), gated on a
same-input replay against N64, then 32 frozen IDs on all four ranks,
then both timing orders.

Target gaps at the current best timed stack (Xe2 gate/up + Xe2 down +
DecodeM8 + socket fan-out, 901.88 prefill / 53.25 decode tok/s):
**3,098.12 prefill** and **86.75 decode tok/s**.

#### Xe2 tail chunk: implemented, three attempts, NOT qualified

The profile above named the 658-row tail as the largest recoverable cost, so
the Xe2 FP32 path was extended to it. Implemented: `launch_signed_gate_up_tail`
and `launch_signed_down_tail` in the isolated bench instantiate the same
pinned kernel with the lane's own `A_avg_M <= 128` branch
(`w4a16_policy_m_32`, M32/N64/K32, 6,580 allocated route rows), asserted at
compile time against `slots/experts > 128` for the M128 instantiation;
`GRIMOIRE_QFN_XE2_TAIL=1` adds the tail dispatch while the control arm stays
byte-identical to the qualified 2,048-only path; a tile-padded tail scratch
(6,656 rows = 6,580 rounded to whole 128-row tiles) was added because the
tail D buffer is not a multiple of the tile; and replay mode
`GRIMOIRE_QFN_XE2_ROUTED_REPLAY=tail` restricts the existing
identical-input N64 comparison to the 658-row chunk, checking each live
slot's BF16 SwiGLU output and the full weighted output.

**No run reached parity, so there are no tail timings and no recovered
milliseconds.** Three separate GPU attempts, each after a clean build
(`build.exit=0`):

| Attempt | Tail kernel | Outcome |
| --- | --- | --- |
| `act-xe2-tail-replay-ctsm0Y` | M32, direct buffer | Xe fault: `Fault response: Unsuccessful -ENOENT`, `Engine memory CAT error [18]: class=ccs`, `Timedout job`, coredump on two cards |
| `act-xe2-tail-replay-ZcZyGi` | qualified M128, direct buffer | **hang**: `stage=tail-replay-rank-2 exit=124` at the 150 s GPU timeout, no fault line |
| `act-xe2-tail-replay-T7MJLN` | qualified M128, padded scratch | Xe fault again (`-ENOENT`), zero replay lines on every rank |

Every attempt left the four devices `normal` with no render node held, and
the serving SHA-256 stayed
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.
The 2,048-row chunk completed in each run before the failure.

**Cause not established.** A D-buffer overrun was the working hypothesis
(the tail allocates exactly 6,580 rows, not a tile multiple); the padded
test ran the tail again and faulted, so that hypothesis is **refuted**, not
merely unproven. The two distinct symptoms (fault with M32 and with padded
M128, hang with direct M128) mean the instantiation, the extent and the
scheduling are each still open. The tail path stays off by default
(`GRIMOIRE_QFN_XE2_TAIL` unset) and every recorded rate still uses the N64
tail. The new translation unit was re-gated on the unchanged qualified
path: `act-xe2-fp32-sDSG4C` finished `stage=verify exit=0` with both arms
(`n64-gate`, `xe2-fp32-gate`) matching **32/32 frozen IDs on all four
ranks**, so the added code is flag-gated and the control behaviour is
unchanged rather than merely assumed. Next step is to isolate the tail
away from TP4: the same 6,580-row shape through `run-single.sh` on one GPU
with no replay and no transport, which separates a kernel/layout defect
from anything the four-rank path contributes. Nobody should quote a tail
gain until that passes and the frozen IDs hold.

#### Tail isolation on one GPU: the M32 kernel itself works

The named isolation was run, bounded. `regressions/.../single_rank.cpp` gained
`--tail-probe`: real layer-0 rank-1 gate/up weights, 6,580 route rows
(~51-52 per expert over 128), the **M32 tail instantiation**, an exactly
6,580-row FP32 destination, one visible GPU, **no N64 comparison, no replay,
no transport**.

`diagnostics/act-xe2-tail-probe-wibN11` (`build.exit=0`, `job.exit=0`):

```
tail probe: rows=6580 capacity=6580 policy=M32/N64/K32 experts=128
tail probe: completed in 73.817 ms, first=-0.024208546 finite=1
```

No xe fault, no `CAT error`, no `Timedout job`, no new dmesg lines, all four
devices `normal`, and the serving SHA-256 still
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`. The
73.817 ms includes first-submit code generation, so it is a liveness and
sanity figure, **not** a steady-state rate and not a tail gain.

**This refutes the reading that the M32 instantiation or its layout is
broken at this shape.** The four-rank failures are therefore caused by
something the four-rank path adds. Two variables are still untested:

1. the tail **down** projection (`K=640, N=2560`, same policy) — the probe
   covered gate/up only, and every four-rank run exercises both;
2. the four-rank integration itself (USM traffic on four cards, the replay's
   extra allocations and its N64 reference, and the fan-out reductions
   interleaved with the tail chunk).

Because the isolation passed, the conditional pivot was **not** taken: the
plan was to abandon this shape only on a second single-GPU failure. The tail
stays off by default (`GRIMOIRE_QFN_XE2_TAIL` unset) and every recorded rate
still uses the N64 tail. Next bounded step is the same probe for the tail
down shape, then a four-rank run with the replay disabled to separate (2)
from the replay machinery.

#### Tail isolation, part 2: both kernels clean alone, four-rank integration faults

**Tail down probe (`K=640, N=2560`, M32/N64/K32), one GPU, no transport.**
`single_rank.cpp` gained `--tail-down-probe`, which stages the real layer-0
rank-1 down weights (packed `{I/8, H}`, zeros `{I/group, H/8}`, scales
`{I/group, H}`, the loader's `^0x88` fold and `0xff` marker), uses the same
6,580-route histogram and an exactly 6,580-row FP32 destination.
`diagnostics/act-xe2-tail-down-run-vI4dK6` (`probe_exit=0`):

```
tail down probe: rows=6580 capacity=6580 K=640 N=2560 policy=M32/N64/K32 experts=128
tail down probe: completed in 72.428 ms, first=-0.1752826 finite=1
```

No fault, no `CAT error`, no `Timedout job`, no new dmesg, and the serving
SHA-256 unchanged. Both tail projections therefore run clean at the real
tail capacity on one GPU (gate/up 73.817 ms, down 72.428 ms — first-submit
figures, not rates).

**Four-rank integration with the N64 replay disabled.** New `tail-gate`
mode: candidate 6 only (`GRIMOIRE_QFN_XE2_TAIL=1`), trace on, Xe2 FP32
gate/up + down, socket fan-out, and **no** `GRIMOIRE_QFN_XE2_ROUTED_REPLAY`,
so no replay buffers, no N64 reference and no interleaved reference passes.
`diagnostics/act-xe2-tail-gate-FgGyK1`: `stage=tail-gate-driver-or-monitor
exit=90`. Rank 0 logged the 2,048-row chunk and its markers, then the run
died: `Fault response: Unsuccessful -ENOENT` on `0000:0a:00.0` plus
`Timedout job` on all four cards. Zero `QFN Xe2 tail gate/up` markers, zero
`QFN Xe2 tail down` markers, no frozen-ID files on any rank, all four
devices `normal` afterwards, serving SHA-256 unchanged.

So the fault is **not** the replay machinery — disabling it reproduces it.
It is also not the kernel in isolation. What remains between them is the
four-rank context: four concurrent cards running this instantiation, the
fan-out reductions around the tail chunk, and the tail's padded scratch
allocations. The tail therefore stays **off** (`GRIMOIRE_QFN_XE2_TAIL`
unset; only candidate 6 sets it), not marked rejected, and no rate on
record uses it. Next diagnostic, if wanted: the same `tail-gate` arm with
the tail restricted to **gate/up only** (down left on N64) and then the
mirror case, to see which of the two instantiations the four-rank path
cannot run.

#### Prefill policy: adopting the lane's A_avg_M rule (m32)

**Premise correction first.** Whole-expert EP4 with 128 of 512 routed
experts resident per rank, gate/up **and** down through the lane's Xe2
W4A16 kernels, is already the baseline and was verified bitwise at
`act-xe2-routed-k7UaWd` (experts `[0,128)`, `[128,256)`, `[256,384)`,
`[384,512)`, full 640-wide projections, all four ranks). Re-implementing it
would compare the baseline to itself, so that is not what was built.

The real deviation from the lane at this shape is the **policy**.
`launch_lane` pinned `w4a16_policy` (M128/N256), justified by *capacity*
(20,480/128 = 160). The lane picks by *live* rows: `A_avg_M = A_total_M /
num_experts`. Measured here, prefill live rows per rank are **mean 5,120,
range 2,313–9,502**, i.e. **A_avg_M 18–74** — every layer inside the lane's
`<= 128` branch, which selects **`w4a16_policy_m_32` (M32/N64)**. So prefill
was running 128-row tiles over ~40 live rows per expert.

Implemented in the isolated qgate candidate: `launch_signed_gate_up_p32` and
`launch_signed_down_p32` (same pinned kernel, `w4a16_policy_m_32`, capacity
`slots`), selected by `GRIMOIRE_QFN_XE2_PREFILL_POLICY=m32` for the full
2,048-row chunk; gate/up, routing and socket fan-out are otherwise
unchanged, and a separate `QFN Xe2 prefill policy: rank=r policy=m32|m128`
line records the choice without disturbing the existing markers.

**Gate: passed, and arithmetically exact.** `diagnostics/act-xe2-policy-ab-WxDBLZ`
(`stage=verify exit=0`) first ran `policy-replay`, which executes N64 and the
m32 Xe2 path on the identical real input: **48/48 layers on all four ranks
bitwise equal**, `stage=down` (real live rows, 2,313–9,502) and
`stage=weighted` (2,048 rows) alike, with `policy=m32` logged per rank. All
four ranks then matched **32/32 frozen IDs**. All five arms recorded and
verified the frozen serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: m128 prefill | 3,008.319 | 899.51 | 651.457 | 47.59 |
| 1: m32 prefill | 2,959.567 | 914.32 | 644.978 | 48.06 |
| 2: m32 prefill | 2,958.815 | 914.56 | 631.016 | 49.13 |
| 2: m128 prefill | 2,984.261 | 906.76 | 626.941 | 49.45 |

m32 is faster in both orders: **−48.752 ms (1.62 %)** and **−25.446 ms
(0.85 %)**. Means: **m32 914.44 prefill / 48.59 decode** against **m128
903.13 / 48.52**. The only kernels this changes are the routed gate/up and
down, so the recovered routed-MoE time is that prefill delta — **≈37 ms
mean** (routed device work would go from 543 ms to ≈506 ms); no re-profile
was taken, so the split is attributed, not re-measured. Decode is untouched
by this change and its movement is noise.

Remaining gaps at this stack: **3,085.56 prefill** and **91.41 decode
tok/s**. Opt-in, one paired A/B, needs repetition before promotion.

**What this does *not* explain, stated as arithmetic.** Even zeroing the
entire routed MoE (543 ms of routed device work, or 989 ms inclusive of its
reduction) would take prefill from 3,001 ms to ≈2,458 ms (≈1,100 tok/s) or
≈2,012 ms (≈1,345 tok/s) — nowhere near the lane's 4.6–5.2 K, and not 4,000.
The 4,000 target therefore requires the **dense path** to shrink: at the same
profile, all-connections 733 ms, reduction transport 543 ms, DeltaNet 499 ms,
full attention 410 ms and PLE 36 ms, none of which this change touches.
Routed-MoE work cannot close this gap by any factor, which is why the tail
(658 of 2,706 rows) was also abandoned.

#### Dense path: HC-down tiling qualified, HC fusion provisional

**Transport note first.** oneCCL ordered was **not** stacked here. It is not
part of the current best stack: `act-ccl-stack-6Bc0jb` measured it slower in
decode in both orders (+4.581 ms, +33.014 ms) with total wall inside noise,
and fan-out is the recorded transport baseline. Adding it would re-introduce
a measured loss into a dense-path measurement.

Stack under test: socket fan-out, Xe2 FP32 gate/up + down, `m32` prefill
policy, chunk 2048, K-prefetch off. `GRIMOIRE_QFN_HC_DOWN_TILED=1` is wired
as candidate 9 in the runner, with candidate 8 as the identical control
differing only in that variable.

**A/B 1 — HC-down tiling (qualified).** `diagnostics/act-xe2-hc-ab-BVh43h`,
`stage=verify exit=0`. Both trace arms matched 32/32 frozen IDs on all four
ranks, and the tiled arm proved engagement from the `GRIMOIRE_BUDGET_DEBUG`
counter: `QFN HC prefill tiled rank=r pos=0 rows=2048 calls=96` and
`pos=2048 rows=658 calls=96` on every rank.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: m32 control | 3,033.798 | 891.95 | 665.587 | 46.58 |
| 1: m32 + HC tiled | 2,859.151 | 946.43 | 633.983 | 48.90 |
| 2: m32 + HC tiled | 2,845.086 | 951.11 | 629.249 | 49.27 |
| 2: m32 control | 3,019.759 | 896.10 | 665.297 | 46.60 |

**−174.647 ms** and **−174.673 ms** — the two orders agree to 0.03 ms, which
is the strongest paired result on this stack so far. Means **948.77 prefill /
49.09 decode** against **894.03 / 46.59**. Recovered **≈174.7 ms** of the
406 ms HC-down component, i.e. all-connections 733 ms → ≈558 ms. This
retires the "opt-in pending repeat" status: HC-down tiling is qualified for
the qgate stack. Decode is not touched by this code path
(`tiled_prefill_hc` is prefill-only), so its movement is not credited.

**A/B 2 — HC fusion on top (provisional).** `diagnostics/act-xe2-hc2-ab-1jKc7r`,
`stage=verify exit=0`, control = m32 + tiled, candidate adds
`GRIMOIRE_QFN_HC_FUSED_PREFILL=1`. All five arms matched 32/32 IDs on all
four ranks.

| Pair / order | Prefill ms | Prefill tok/s |
| --- | ---: | ---: |
| 1: m32 + tiled | 2,904.134 | 931.78 |
| 1: + HC fused | 2,818.214 | 960.18 |
| 2: + HC fused | 2,811.539 | 962.46 |
| 2: m32 + tiled | 2,835.357 | 954.38 |

−85.920 ms (2.96 %) and −23.818 ms (0.84 %): same sign, but the smaller pair
delta is inside the ~17 ms spread between this control and A/B 1's tiled arm.
Means **961.32** vs **943.08** prefill. Recorded as provisional: a third
order-reversed pair is required before it counts as promotion.

Attributed HC total afterwards: 733 ms → ≈558 ms (tiled) → ≈503 ms
(tiled + fused). Every arm in both runs recorded and verified the frozen
serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

Target gaps, relative to 4,000 prefill / 140 decode:

| Stack | Prefill tok/s | Gap | Decode tok/s | Gap |
| --- | ---: | ---: | ---: | ---: |
| m32 + HC tiled (qualified) | 948.77 | **3,051.23** | 49.09 | **90.91** |
| + HC fused (provisional) | 961.32 | 3,038.68 | 48.80 | 91.20 |

The earlier estimates for this stack were 3,085.56 / 91.41, so HC-down tiling
recovered about 34 tok/s of the prefill gap. Decode is unchanged in code by
either HC change; the best decode configuration on record is still Xe2
DecodeM8 (53.25 tok/s), and combining it with this prefill stack is untested.

#### Decode: DecodeM8 combined with the dense-path stack

Candidate 11 = m32 prefill policy + HC-down tiled + DecodeM8, against
candidate 9 (identical minus DecodeM8). `diagnostics/act-xe2-m8-combo-ab-p4GQLV`,
`stage=verify exit=0`. The gate arm matched **32/32 frozen IDs on all four
ranks** and proved M8 executed: **31 `QFN M8 decode:` lines per rank**, each
`layers=48 mode=candidate`.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: m32 + tiled | 2,857.927 | 946.84 | 645.957 | 47.99 |
| 1: + DecodeM8 | 2,863.270 | 945.07 | 579.683 | 53.48 |
| 2: + DecodeM8 | 2,825.268 | 957.79 | 573.196 | 54.08 |
| 2: m32 + tiled | 2,895.106 | 934.68 | 667.991 | 46.41 |

Decode **−66.274 ms (10.26 %)** and **−94.795 ms (14.19 %)** — both orders,
far outside the ~2 tok/s run-to-run decode spread. Means **53.78 decode /
951.43 prefill** against **47.20 / 940.76**. Prefill is untouched by M8
(decode-only path); its 11 tok/s difference is noise and not credited.
Decode gap: **140 − 53.78 = 86.22** tok/s; prefill gap **3,048.57**.

**Reproducibility trap recorded.** The first attempt at this combination was
inert: `m8-combo-ab` had not been added to the overlay-chain selection, so
DecodeM8 was never compiled into that binary and the "candidate" arm was the
baseline carrying an unused env var — its IDs passed trivially and no M8
marker appeared. The absence of `QFN M8 decode:` lines is the tell; a run
without them is not an M8 measurement.

#### Decode cost after the combination, and the transport attempt

Fresh fenced profile on candidate 11 (`profile-combo` in the same run; means
of four ranks, diagnostics only):

| Phase | Prefill ms | Decode ms |
| --- | ---: | ---: |
| wall | 2,960.8 | 689.6 |
| MoE inclusive / excluding its reduction | 961.0 / 685.1 | 248.5 / 189.0 |
| routed local GEMM device work | 514.0 | not instrumented under M8 |
| all connections (HC) | 655.3 | 176.2 |
| — HC down | 326.3 | — |
| reduction transport | 585.0 | 141.7 |
| DeltaNet excluding transport | 500.0 | 94.3 |
| full attention excluding transport | 409.8 | 48.4 |
| PLE excluding transport | 48.8 | 14.3 |
| head | — | 21.0 |

Two corrections to the older reading. HC-down tiling shows up here as
**326.3 ms** against the earlier 406.1, matching the ≈175 ms A/B recovery
when both chunks and the 2,048-only scope are accounted for. And
transport is **no longer the largest decode cost**: it is third at 141.7 ms
behind MoE excluding its reduction (189.0 ms, now mostly M8 plus the shared
expert) and HC (176.2 ms). The `act-decode-projections-MGaXLt` split that
named transport first predates DecodeM8 and the tiling.

**Transport attack, gated and not established.** The only implemented
decode-transport candidate is the control-packet fan-out
(`GRIMOIRE_QFN_TP_DECODE_FANOUT=1`), previously measured at −38 ms and never
shipped. On top of M8 (`diagnostics/act-xe2-dfan-ab-Xn0vd4`,
`stage=verify exit=0`, all six arms 32/32 IDs):

| Pair / order | Decode ms | Decode tok/s |
| --- | ---: | ---: |
| 1: m32 + tiled + M8 | 599.568 | 51.70 |
| 1: + decode fan-out | 581.206 | 53.34 |
| 2: + decode fan-out | 575.851 | 53.83 |
| 2: m32 + tiled + M8 | 573.646 | 54.04 |

−18.362 ms in pair 1, **+2.205 ms (control faster)** in pair 2. Means
**53.59 vs 52.87** — a 0.7 tok/s difference, smaller than the ~2 tok/s
spread between these runs of the same control arm. **Not established**, stays
opt-in, and needs several more pairs before it can be called a win. oneCCL
remains excluded for the reason already on record.

Current best decode on this stack is therefore **DecodeM8 at 53.78 tok/s
(gap 86.22)**, and the next real decode target is the 189.0 ms of MoE time
excluding its reduction — not transport.

#### Whole-expert EP4: verified as the live baseline, not implemented again

The task asked for whole-expert EP4 placement (128 of 512 experts resident
per rank, gate/up **and** down through the lane's Xe2 W4A16 kernels). That is
already what this stack does, so it was verified from artifacts instead of
re-implemented. Evidence, all from this tree and existing runs:

- **Sharding:** `include/b70/qfn_shard_plan.hpp:101` —
  `routed_experts{r * (s.routed_experts / world), s.routed_experts / world}`
  = 512/4 contiguous experts per rank.
- **Loaded placement, printed by an actual four-rank run**
  (`act-xe2-routed-k7UaWd`): `QFN TP4 rank=0 validated 48 layers: experts
  [0,128)`, rank 1 `[128,256)`, rank 2 `[256,384)`, rank 3 `[384,512)`.
- **Complete weights per local expert:** `src/grimoire.cpp:8775-8784` takes
  `expert_begin`/`expert_count` from that plan and allocates
  `gu_rows = expert_count * 2 * I`, `dn_rows = expert_count * H` — full
  640-wide gate/up/down, no width slicing.
- **Both projections on the lane's Xe2 kernels, on the live stack**
  (`act-xe2-m8-combo-ab-p4GQLV`, ranks 0 and 3):
  `QFN Xe2 gate/up: rank=0 rows=2048 calls=48 output=fp32` **and**
  `QFN Xe2 down: rank=0 rows=2048 calls=48 output=fp32`, with
  `policy=m32`.
- **Arithmetic signature of EP4:** mean live routed rows are **5,120 per
  rank per layer** over 192 layer-ranks — exactly 20,480 token-slots ÷ 4
  ranks, which only happens when experts are split whole across ranks.
- **Exactness on those rows:** `act-xe2-routed-k7UaWd` and
  `act-xe2-down-WH5q2J` each report 48/48 `stage=down` and 48/48
  `stage=weighted` bitwise-equal layers per rank (down rows summing 225,302
  over 48 layers, weighted 98,304 = 2,048×48).

No new EP4 work was performed, because running it against itself would not
be an experiment. **The arithmetic also rules it out as the gap closer:**
zeroing routed MoE entirely (514 ms of routed device work, 685 ms of MoE
excluding its reduction) moves prefill from ≈2,960 ms to ≈2,446 / 2,275 ms,
i.e. ≈1,105 / 1,190 tok/s. A 4.2× gap cannot be closed by work inside MoE,
and the profile above shows the rest: HC 655 ms, transport 585 ms, DeltaNet
500 ms, attention 410 ms.

#### HC fusion: third pair taken, qualified

Candidate 10/13 (`GRIMOIRE_QFN_HC_FUSED_PREFILL=1`) had two pairs on record
(−85.920 ms, −23.818 ms) and the second was inside the arm spread. It now
has a third order-reversed pair **on the M8-complete stack**:
`diagnostics/act-xe2-fuse3-ab-MAz4OS`, `stage=verify exit=0`, gate arm with
**31 `QFN M8 decode:` lines per rank** and 32/32 IDs on all four ranks.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: m32 + tiled + M8 | 2,844.951 | 951.16 | 595.585 | 52.05 |
| 1: + HC fused | 2,806.627 | 964.15 | 573.394 | 54.06 |
| 2: + HC fused | 2,803.588 | 965.19 | 589.435 | 52.59 |
| 2: m32 + tiled + M8 | 2,838.387 | 953.36 | 587.101 | 52.80 |

**−38.324 ms (1.35 %)** and **−34.799 ms (1.23 %)** — consistent sign and
similar magnitude. Across all three pairs every order reversal favours
fusion, so it graduates from provisional to **qualified** for the qgate
stack. Means **964.67 prefill / 53.33 decode** against **952.26 / 52.42**.
HC fusion is a prefill-only path, so the decode movement is not credited.

Best measured stack and gaps (targets 4,000 prefill / 140 decode):

| Stack | Prefill tok/s | Gap | Decode tok/s | Gap |
| --- | ---: | ---: | ---: | ---: |
| m32 + HC tiled + HC fused + DecodeM8 | 964.67 | **3,035.33** | 53.33 | **86.67** |
| best decode measured separately (DecodeM8, no fusion run) | — | — | 53.78 | 86.22 |

Prefill has moved 903.13 → 948.77 (m32 + tiling) → 964.67 (fusion) since the
policy finding. Every arm in this run recorded and verified the frozen
serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

#### DeltaNet prefill breakdown: the recurrent scan is 4 %, not the cost

DeltaNet had no working sub-phase timing: `traced()` in the prefill path is
inert below `tp_trace` (the one-token TP4 trace), so only the whole phase was
visible. `regressions/2026-11-20-qfn-xe2-gate-up/dn-profile.awk` adds
`GRIMOIRE_QFN_DN_PROFILE`, a fenced profiler over the prefill LINEAR_ATTN
path with five buckets: qkv projection, conv+norm, gates, recurrent scan,
z projection. Diagnostics only — every lap forces a queue wait.

**Instrumentation trap worth keeping.** The first version laid the first lap
immediately after `mm(d.la_qkv)`, so bucket 0 absorbed everything since the
previous linear layer — three non-linear layers of HC, MoE and transport —
and reported `qkv_ms=1510`. That number is not a projection (it exceeds the
whole DeltaNet phase). A lap closing that region before the qkv call
(`dnp_lap(5)`) makes the buckets honest; the corrected run reports
`qkv_ms=168`.

`diagnostics/act-xe2-dn-prof-FEKp5p` (arm `dn-profile`, trace off, 32/32
frozen IDs on all four ranks, serving SHA-256 verified). Per rank, summed
over the twelve linear layers in each chunk:

| Stage | 2,048-row chunk | 658-row chunk | Total |
| --- | ---: | ---: | ---: |
| qkv projection | 168.4 ms | 23.2 ms | **191.6 ms** |
| conv + q/k norms | 104.9 ms | 14.0 ms | **118.9 ms** |
| z projection | 49.6 ms | 21.0 ms | 70.6 ms |
| recurrent scan (`launch_deltanet_prefill`) | 14.3 ms | 4.8 ms | **19.1 ms** |
| gates | 0.65 ms | 0.73 ms | 1.4 ms |
| DeltaNet own cost | 337.7 ms | 63.7 ms | ≈402 ms |

Ranks agree to within 1.5 % (own cost 401.4 / 402.0 / 403.7 / 406.1 ms), and
402 ms against the 500 ms phase figure leaves ≈100 ms for the trailing
norm-gate and the layer reduction.

**Two conclusions.** The **recurrent scan is only 19 ms (4 %)** — the obvious
target was the wrong one, and optimising it could not pay for itself. The
largest DeltaNet cost is the **qkv projection at 191.6 ms**, and it is
suspicious on its own terms: 12 layers × 14 ms for a 2,048×12,288×2,560
projection is ≈0.77 TFLOP/s per rank, one to two orders below what these
cards sustain. That points at `mm(d.la_qkv, …)` not taking an efficient
tiled path for N=12,288 rather than at anything DeltaNet-specific, and it is
where the next attempt belongs — comparing its rate against the MoE gate/up
GEMM rate on the same stack before changing anything.

**Not implemented yet.** No DeltaNet optimisation was written this pass; the
measurement above is the deliverable, plus the finding that the intended
target (the scan) is not the cost. The `dn-prof` mode also still falls
through into the default A/B after its arm (missing `exit`), so its receipt
is not clean — `stage=1-xe2-parity-*` rather than `stage=verify` — though the
arm's own logs and ID files are intact and were verified separately.

#### Projection dispatch: `gemm_fast` is the slow path at large M (and why the naive cap is not the fix)

**First, a retraction.** The previous entry claimed the qkv projection runs
at "≈0.77 TFLOP/s, one to two orders below these cards". That was
**arithmetic error**: it divided the total 168.4 ms by per-layer FLOPs.
Recomputed: 2,048×12,288×2,560 = 128.8 GFLOP per layer, ×12 = 1.546 TFLOP in
168.4 ms = **9.18 TFLOP/s per rank**. The stack's best measured kernel (the
Xe2 W4A16 gate/up at 1.167 ms for 4,354×1,280×2,560) runs at
**24.45 TFLOP/s**, so the real headroom is **2.66×**, not 10×.

Also, N=12,288 does **not** fall out of `gemm_fast`: `np * K` = 31.5 M is
under the 512 M `kMaxScratchElems` cap, so `gemm_fast_supported()` returns
true and the projection takes the dequant-once-then-GEMM path. There was no
threshold miss to repair.

**What the dispatch actually costs, measured.** `GRIMOIRE_NO_FAST_GEMM`
disables that path for the whole stack. `diagnostics/act-xe2-dn-ab-zm650X`,
`stage=verify exit=0`, gate arm 32/32 IDs on all four ranks, every arm's
serving SHA-256 verified:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: fast path on (legacy) | 2,876.636 | 940.68 | 53.39 |
| 1: fast path off | 2,671.574 | 1,012.89 | 50.19 |
| 2: fast path off | 2,620.149 | 1,032.77 | 52.29 |
| 2: fast path on (legacy) | 2,813.827 | 961.68 | 54.23 |

**−205.062 ms and −193.678 ms** in the two orders. Means **1,022.83 prefill /
51.24 decode** against **951.18 / 53.81**. `gemm_fast` dequantises the whole
weight into scratch on every call — fine when M is small, dominated by that
traffic at M=658/2,048 — so it is a real dispatch defect, worth **+71.7
tok/s prefill (7.5 %)**, with a **−2.6 tok/s decode** cost.

**A naive M cap does not capture it.** An `M <= 512` default in
`gemm_fast_supported` (`diagnostics/act-xe2-gfmax-ab-DO2VG6`, `stage=verify
exit=0`, all gate IDs matched) gave only **−36.523 / −72.357 ms**: legacy
941.07 → 959.24 prefill. The reason is diagnostic — **`gemm_fast_supported`
also gates the qualified HC-down tiling**, so the cap silently disabled that
175 ms win, and the net stack (959.24) landed *below* the best known 964.67.
The remaining `gemm_fast` users are not behind that predicate, which is why
the cap recovered only a quarter of the full-disable gain.

**Left as a knob, not a default.** `GRIMOIRE_FAST_GEMM_MAX_M` now exists
(default unbounded, i.e. legacy behaviour restored) so the finding is
reproducible without leaving the tree slower. The correct fix is narrower:
gate `gemm_fast` by *caller* — the large-M projection call sites — and give
HC-down tiling its own eligibility check instead of sharing
`gemm_fast_supported`. That should capture ≈+70 tok/s prefill without the
decode regression, and is the next item.

Best stack therefore remains m32 + HC tiled + HC fused + DecodeM8 at
**964.67 prefill / 53.33 decode**, gaps **3,035.33 / 86.67**. Recovered qkv
ms this pass: **0** — the dispatch defect is identified and quantified, but
not yet shipped, because every form I could verify either cost decode or
lost the HC tiling.

**Runner fix requested**: `dn-prof` now ends with `stage=verify`,
`sha256sum -c` and `exit` instead of falling through into the default A/B
(it lives in `run-candidate.sh`, not `model-gate.sh`). It is applied and
syntax-checked; the two runs above both report `stage=verify exit=0`.

**Shipped: per-caller eligibility.** `gemm_fast_supported()` is now split.
`gemm_fast_shape_ok(w, M)` keeps the shape/scratch test (`!off && M >= 32 &&
payload && N%16==0 && K%KC1==0 && np*K <= kMaxScratchElems`) with no upper
bound, and `gemm_fast_supported()` adds the cap (`M <= 512` by default,
`GRIMOIRE_FAST_GEMM_MAX_M` overrides). The HC-down tiled projection uses the
shape predicate in **both** places it needs it — the `grimoire.cpp` call site
and its own self-check inside `launch_qfn_hc_down_tiled` — so an M cap chosen
for the dense projections can no longer switch that 175 ms path off.

One build detail mattered: the runner reuses base objects verbatim and only
recompiled the overlay translation unit, so the source change was invisible
until `src/gemm_fast.cpp` was added to that per-run compile list. The first
attempt at this A/B would otherwise have measured an unchanged binary.

`diagnostics/act-xe2-gfmax-ab-LNlYls`, `stage=verify exit=0`, gate arm 32/32
frozen IDs on all four ranks, serving SHA-256 verified on all five arms:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: legacy dispatch | 2,945.384 | 918.73 | 53.26 |
| 1: per-caller cap | 2,576.837 | 1,050.12 | 52.68 |
| 2: per-caller cap | 2,544.795 | 1,063.35 | 53.09 |
| 2: legacy dispatch | 2,813.367 | 961.84 | 53.65 |

**−368.547 ms (12.51 %)** and **−268.572 ms (9.55 %)**, means **1,056.73
prefill / 52.89 decode** against **940.29 / 53.45**. Recovered ≈ **318.6 ms**
of prefill. Decode is within noise of the control (−0.56 tok/s), unlike the
blanket disable which cost −2.6 tok/s, and unlike the previous capped
attempt this one is *above* the earlier best stack (1,056.73 vs 964.67)
because HC-down tiling keeps running. No `GRIMOIRE_BUDGET_DEBUG` marker was
captured in this run, so the tiling restoration is inferred from that
magnitude rather than logged per rank — worth adding to the gate arm next
time.

Target gaps, now: **4,000 − 1,056.73 = 2,943.27 prefill** and **140 − 52.89 =
87.11 decode** (best decode measured separately is 53.78, gap 86.22).

**Next remaining bottleneck.** With prefill at 2,561 ms and MoE's routed
device work at 514 ms, the profile's remaining order is HC ≈480 ms (after
tiling and fusion), reduction transport 585 ms, DeltaNet ≈400 ms (qkv 192,
conv+norm 119, z 71, scan 19) and full attention 410 ms. Transport is now
the single largest, and the only implemented alternative (oneCCL) is on
record as losing decode in both orders — so the honest next item is either a
new transport overlap or the attention/DeltaNet conv paths, not another
dispatch threshold.

#### HC tiling observed, and why the transport overlap has no headroom

**1. The tiling restoration is now observed, not inferred.**
`diagnostics/act-xe2-percaller-check-GDW2rZ` ran the per-caller-cap stack with
`GRIMOIRE_BUDGET_DEBUG=1`. Every rank logged both markers at both chunks:

```
QFN HC prefill tiled rank=0 pos=0    rows=2048 calls=96
QFN HC prefill fused rank=0 pos=0    rows=2048 calls=94
QFN HC prefill tiled rank=0 pos=2048 rows=658  calls=96
QFN HC prefill fused rank=0 pos=2048 rows=658  calls=94
```

…identically for ranks 1, 2 and 3 (16 tiled and 16 fused lines in total), so
the ≈318 ms attributed to restoring that path in the previous entry is a
measured restoration, not a magnitude guess. Caveat on the receipt: the arm
ends `stage=percaller-gate-parity-rank-1 exit=1` because `percaller-check`
had fallen into the `else` build branch and so had no decode overlay — my
marker check demanded the 31 `QFN M8 decode:` lines that branch could not
emit. DecodeM8 is decode-only, so the prefill tiling evidence is unaffected;
the build branch is fixed for future runs.

**2. The requested overlap cannot help, and the drain number says why.**
Prefill reductions, from the fresh profile (`profile-m8`, rank 0):

| Quantity | Value |
| --- | ---: |
| reduction calls per prompt | 194 |
| payload moved per rank | 2.69 GB |
| transport time | 569.3 ms |
| **queue drain before the reduction** | **0.562 ms (0.099 %)** |
| effective rate | 4.72 GB/s |
| per call | 2.935 ms for 13.21 MiB |

Issuing the exchange on a second queue and waiting at first consumption can
only hide *idle* time, and there is none: the drain is **0.1 %** of the
reduction cost. So that restructuring was **not implemented** — it would add
cross-queue ordering risk for at most ~0.6 ms. The premise that the 585 ms is
mostly waiting does not survive the measurement; the time is the exchange
itself.

**3. The real transport anomaly.** The standalone peer-push microbenchmark
moves 5 MiB in 192 µs = **27.3 GB/s**, while the in-model reduction averages
**4.72 GB/s** — **5.8× slower**. At the standalone rate the same 2.69 GB would
take ≈98 ms instead of 569 ms, i.e. **≈471 ms recoverable**, which is now the
largest single opportunity on record and larger than everything the dense
path has yielded so far. Why the in-model rate is 5.8× lower is **not
established**: candidates are the per-call payload shape (13.21 MiB average vs
the benchmark's fixed 5 MiB), serialisation around each reduction in the
in-order queue, or the fan-out path's per-call setup. Diagnosing that is the
next item; a second queue is not it.

Best stack and gaps are unchanged by this pass — m32 + HC tiled + HC fused +
DecodeM8 + per-caller `gemm_fast` cap: **1,056.73 prefill / 52.89 decode**,
gaps **2,943.27 / 87.11**.

#### Transport anomaly: the handshake is not the cause (negative result)

**Where the 5.8× actually sits, per chunk.** Decomposing the reduction by chunk
(rank 0, `act-xe2-profile-FFTpyL`):

| Chunk | calls | payload/call | time/call | rate |
| --- | ---: | ---: | ---: | ---: |
| 2,048 rows | 97 | 20.97 MB | 2.05 ms | **10.2 GB/s** |
| 658 rows | 97 | 6.74 MB | 1.76 ms | **3.8 GB/s** |
| total | 194 | 2.69 GB | — | 4.72 GB/s |

The small chunk is 2.7× less efficient per byte than the large one, which is
the signature of fixed per-call cost, and the large chunk is still 2.7× under
the 27.3 GB/s standalone peer-push. Both effects are real; only one was
testable cheaply.

**What was implemented.** `GRIMOIRE_QFN_TP_SYNC=fanout` sets `qfn_all_fanout`,
which was OR-ed into *every* reduction, so each prefill call also paid the
`exchange_small_fanout` control handshake — three socket round-trips, written
for the 10 KB decode reduction. `tp_block_reduce` now bounds it by payload
size (`GRIMOIRE_QFN_FANOUT_MAX_BYTES`, default 64 KiB), so prefill takes the
plain per-link exchange and decode keeps the handshake. The reduction and its
`((r0+r1)+r2)+r3` sum order are untouched: only the packet exchange changes.

**Measured: no effect, and a first A/B that I had to throw away.** The first
pair used candidate 13 as control, which also restores the legacy `gemm_fast`
dispatch — **two variables**, and its −263.844 / −259.235 ms was the gemm
dispatch, not the handshake. `diagnostics/act-xe2-fanout-ab-eiyz9a` is reported
only as that flawed pair. The corrected A/B holds the capped gemm dispatch
constant and varies only the size gate (`diagnostics/act-xe2-fanout2-ab-nJv1dL`,
`stage=verify exit=0`, all five arms 32/32 IDs, serving SHA-256 verified):

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: fan-out legacy (control) | 2,563.979 | 1,055.39 | 52.64 |
| 1: fan-out gated | 2,554.970 | 1,059.11 | 53.56 |
| 2: fan-out gated | 2,557.092 | 1,058.23 | 53.85 |
| 2: fan-out legacy | 2,540.413 | 1,065.18 | 53.83 |

−9.009 ms then **+16.679 ms (control faster)**: sign flips, magnitude inside
the arm spread. Means **1,058.67 vs 1,060.29**. **The handshake is not the
cause of the 5.8× gap**, and transport time is unchanged at ≈569 ms. The gate
is kept (it removes 194 unnecessary socket handshakes per prompt and measures
neutral) but it is **not** a win and must not be quoted as one.

**What is left in the transport, untested.** The remaining candidates are the
per-call data-path chunking (`limit` is sized from the link capacity, so a
20.97 MB payload is split across loop iterations), the FP32 sum kernel itself,
and queue serialisation around each `q.wait_and_throw()` in
`PeerPush4::allreduce`. Isolating those needs a prefill-side `Timing`
breakdown (copy/ready/sum/reuse) — today only decode records it — which is the
next concrete step.

**EP4 was not implemented, again, because it is the baseline.** Fourth
verification, from live artifacts: `qfn_shard_plan.hpp:101` gives 512/4 = 128
contiguous experts per rank; a real run prints `validated 48 layers: experts
[0,128) / [128,256) / [256,384) / [384,512)`; `grimoire.cpp:8775-8784` allocates
full 640-wide gate/up/down per local expert; the current stack logs both
`QFN Xe2 gate/up: … output=fp32` and `QFN Xe2 down: … output=fp32`; mean live
routed rows are 5,120/rank/layer = 20,480 ÷ 4. Re-running it would compare the
baseline to itself, and the arithmetic already on record (zeroing routed MoE
entirely leaves ≈1,105 tok/s) rules it out as the route to 4,000.

Best stack and gaps: **1,058.67 prefill / 53.70 decode**, gaps **2,941.33 /
86.30**. Next bottleneck: the transport data path (569 ms, 5.8× off
standalone), then full attention 410 ms and DeltaNet conv+norm 119 ms.

#### Prefill transport decomposed: data movement, not the sum or the chunking

**Instrumentation shipped.** Prefill reductions only had `comm_ms` and
`drain_ms`. The prefill profiler's bucket now carries a `PeerPush4::Timing`
and passes it through, so `QFN TP prefill reduce` also reports
`copy_ms/ready_ms/sum_ms/reuse_ms` per kind and payload — the buckets decode
has had all along.

`diagnostics/act-xe2-tp-prof-9eIIuR` (`stage=verify exit=0`, 32/32 IDs, serving
verified), rank 0, both chunks:

| Component | ms | share of 579 ms |
| --- | ---: | ---: |
| copy | 319.1 | **55 %** |
| ready (peer arrival) | 228.5 | **39 %** |
| FP32 rank-order sum | 25.4 | 4.4 % |
| reuse | 5.8 | 1 % |

Per kind, 2,048-row chunk: deltanet 154.5 ms over 36 calls (copy 88.4, ready
57.5, sum 7.1), attention 49.5 over 12 (copy 31.3, ready 15.5), moe 200.7 over
48 (copy 120.4, ready 68.9, sum 9.3). The 658-row chunk is the same shape at
≈40 % of the bytes.

**This retires two of the three candidates and corrects one earlier claim.**

- **The FP32 sum is 4.4 %** — it cannot be the 5.8× gap, and the
  `((r0+r1)+r2)+r3` order stays exactly as it is.
- **Queue serialisation is not transport time at all.** `drain` reaches 328 ms
  for deltanet in this run, but it is measured *before* the reduction and is
  compute still in flight (the earlier 0.099 % figure came from a run with
  `GRIMOIRE_Q4_TIMING` fences, which had already emptied the queue — the two
  numbers measure different things and neither is idle transport).
- **Bigger chunks are worse.** `limit` derives from the link capacity, so
  `GRIMOIRE_P2P_CHUNK_MB=32` makes a 20.97 MB payload one transfer instead of
  three. `diagnostics/act-xe2-chunk-ab-v9E6y7` (`stage=verify exit=0`, five
  arms, 32/32 IDs, serving verified):

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: 8 MiB chunks | 2,554.950 | 1,059.12 | 52.76 |
| 1: 32 MiB chunks | 2,628.160 | 1,029.62 | 50.60 |
| 2: 32 MiB chunks | 2,590.522 | 1,044.58 | 53.13 |
| 2: 8 MiB chunks | 2,549.813 | 1,061.25 | 54.09 |

**+73.210 ms and +40.709 ms: worse in both orders.** The 8 MiB default stays.

So the 5.8× is squarely in the **copy + peer-arrival path** (548 ms of 579,
~8.4 GB/s for the copy alone against 27.3 GB/s standalone), and it is now
narrowed to the copy loop's own mechanics — peer-mapped `q.memcpy` versus a
DMA/event push, and what `event_push` does per chunk — rather than any
parameter or kernel I can flip. Recovered transport ms this pass: **0**; the
instrumentation and three refutations are the deliverable. The `.reuse` and
explicit `.copy` split also shows only 5.8 ms of buffer recycling, so the
reuse guard is not costing anything either.

Best stack and gaps unchanged: **1,060.18 prefill / 53.42 decode**, gaps
**2,939.82 / 86.58**. Next bottleneck: the copy path above; after that full
attention (410 ms) and DeltaNet conv+norm (119 ms). EP4 remains the baseline
(see the four verifications on record) and is still not the route to 4,000:
zeroing routed MoE entirely leaves ≈1,190 tok/s.

#### Transport: the per-chunk size is the lever (1 MiB), the overlap is not

The breakdown left copy 319 ms + peer-arrival 228.5 ms unfixed. Both were
tested this pass.

**Per-chunk memcpy size, three order-reversed pairs.** `limit` derives from the
link capacity, so `GRIMOIRE_P2P_CHUNK_MB` sets how a 20.97 MB payload is split.
All three runs: `stage=verify exit=0`, gate arm 32/32 IDs on all four ranks,
per-arm serving SHA-256 verified.

| Pair | 8 MiB | 5 MiB | 1 MiB | Delta |
| --- | ---: | ---: | ---: | ---: |
| 5 vs 8 (`p2p5-ab-telLkO`) | 1,043.37 | 1,059.01 | — | pair1 −9.552 ms, pair2 −67.106 ms |
| 1 vs 8 (`p2p1-ab-izndLP`) | 1,055.45 | — | 1,071.85 | pair1 −38.697 ms, pair2 −39.739 ms |
| 1 vs 5 (`p2p15-ab-5SgqNR`) | — | 1,058.88 | 1,066.68 | pair1 −31.160 ms, pair2 −6.262 ms |

(prefill tok/s; every pair is order-reversed, all twelve deltas negative.)
1 MiB ≥ 5 MiB ≥ 8 MiB, and 32 MiB had already measured **+73.210 / +40.709 ms
worse**. So the 27.3 GB/s standalone shape does not transfer — that benchmark
used a single 5 MiB transfer per call on an otherwise idle device, whereas here
each call is one payload among 194 and 1 MiB splits keep the copy granular
enough to interleave with the peer arrivals.

The runner default is now **1 MiB** (`GRIMOIRE_P2P_CHUNK_MB=1`, was 8).
Recovered: ≈**35–39 ms** of prefill against the 8 MiB default — real and
repeatable in every pair, but **1.4 %**, not the ~470 ms the rate comparison
suggested. Transport remains ≈540 ms: the copy path's 8.4 GB/s is not a chunk
size artefact.

**The second-queue overlap was not shipped, and the measurement is why.**
`ready` (228.5 ms) is time waiting for **peer** data to arrive, and all four
ranks run the same schedule, so a rank that finishes its producer early simply
waits longer. Moving our own push to another in-order queue cannot make peers
arrive sooner; it would only shorten the local *drain*, which the breakdown
shows is not inside `comm` at all (it is compute still in flight, and the
reduction is immediately consumed by the residual add in the same layer). A
true overlap needs cross-layer pipelining — starting layer i+1's independent
work while layer i's reduction completes — which is a scheduling restructure,
not a queue annotation, and is the honest next item if transport is pursued
further.

Best stack and gaps: **1,071.85 prefill / 52.74 decode** (1 MiB arm),
gaps **2,928.15 / 87.26**; across the three runs the 1 MiB arm measured
1,066.68–1,071.85 prefill and 52.74–54.03 decode. Next bottleneck: the copy
path itself (8.4 GB/s), then full attention 410 ms and DeltaNet conv+norm
119 ms — both still untouched.

#### Attention: already on the matrix unit with fused softmax (the pattern does not apply)

The instruction was to split the 410 ms into softmax/normalisation versus the
QK^T / PV GEMMs and then remove a separate pass the way HC-down tiling did.
Source answers the first half before any measurement:

- `launch_flash_prefill` (`prefill.cpp:1995`) dispatches on
  `flash_fast_supported(head_dim, num_heads, num_kv_heads)`, which requires
  `head_dim == 128 || head_dim == 256`, `num_kv_heads > 0` and
  `num_heads % num_kv_heads == 0`.
- This model's shard plan refuses anything but **24 Q heads, 2 KV heads,
  width 256**, so all three conditions hold and prefill takes
  `launch_flash_prefill_fast` — `flash_prefill_kernel()` names it
  "ESIMD flash on DPAS" or "joint_matrix flash on XMX".
- In that kernel the softmax is **fused**: "the online softmax on its own S/P
  slice, no work-group barrier in the key loop".

So there is **no separate softmax/normalisation pass to eliminate**, and the
HC-down pattern (output-only tiling to delete a second pass) has nothing to
attach to here. The only genuinely separate stage in this path is the K/V
repack pre-pass (`Kp [KVH][D/2][Sp][2]`, `Vp [KVH][Sp/2][D][2]`, plus the baked
Q scale), which converts the layer's e4m3 cache once per layer.

**Measurement attempt failed and was reverted.** I added a
`GRIMOIRE_FLASH_TIMING` bracket to split that pre-pass from the fused kernel.
The patch was placed by searching for `q.submit(` after the instrumentation
anchor, which matched a *different* function — it corrupted a decode MXFP4
GEMM's `return q.submit(...)` into `return if (flash_timed) { ... }`. The build
failed (`src/gemm_fast.cpp:1462: expected expression`, `stage=build exit=1`,
no GPU work). I reverted both insertions and re-verified the file compiles
clean (`icpx … -c src/gemm_fast.cpp` → `compile_rc=0`, pre-existing warnings
only). **The repack/main split is therefore still unmeasured** — no numbers to
report for it, and I am not going to estimate one.

DeltaNet conv+norm (119 ms) was not attempted in this pass.

Recovered ms: **0**. Best stack unchanged: 1 MiB peer chunk default, prefill
**1,071.85**, decode **52.74**, gaps **2,928.15 / 87.26**. Next bottleneck is
still full attention (410 ms) and then DeltaNet conv+norm (119 ms); the
concrete next step is the same split measurement with the timer placed by
line inside `launch_flash_prefill_fast` itself, which decides whether the
repack pre-pass or the fused QK^T/softmax/PV loop deserves the work.

#### Attention is QSA, not dense flash — and the transport overlap still does not ship

**Correction to the previous entry.** I wrote that prefill attention is "already
on the matrix unit with fused softmax". That is true of
`launch_flash_prefill` / `flash_fast_impl`, which is what I instrumented. It is
**not** the path this model takes: its attention is **QSA sparse block
attention**, and the prefill call site (`grimoire.cpp:20064`) is
`launch_flash_prefill_qsa(qv + rr*QH*HD, kc, vc, …)` preceded by
`launch_qsa_block_bits` block selection. So the fused-softmax argument does not
carry over, and the dense-flash timer I added — correctly placed this time —
never fires for this model.

**The timer placement is now sound.** Anchored by line inside the function
(entry after `int window = 0) {`, prepass print immediately before its
`return q.submit(`), using the real parameter names (`H`, `KVH`, `D`,
`tokens`, `start`), and **compiled with `icpx -c src/gemm_fast.cpp` before any
GPU run**: `compile_rc=0`. That is the process fix for the turn that corrupted
a decode GEMM: no GPU time is spent on an insert that does not build.

`diagnostics/act-xe2-flash-prof-r0Qy8J` then ran `stage=verify exit=0` with
32/32 IDs and **printed no prepass lines at all** — which is how the QSA
discovery was made. The QSA path's own fenced breakdown exists
(`QSA prefill: prep … logits … topk … expand … sparse attn`), but in this
configuration it prints **all zeros**, so no attribution is available from it
either. Attention's 410 ms therefore remains **undecomposed**, now for a known
reason (the instrumentation targets the dense path and the QSA accountant is
inert here), and the next step is to instrument `launch_flash_prefill_qsa`
itself: block-selection (logits/topk/expand) versus sparse attention.

**Transport overlap: still not shipped, and this is the third pass.** The
measured basis still contradicts the premise. `ready` (228.5 ms) is the
`PeerPush4` packet exchange and peer-arrival wait, which in this transport is
**host-side socket work** (`exchange_packet`, `exchange_small_fanout`,
`read_all`) — not a GPU queue ordering. A separate in-order queue can change
when *device* work is issued; it cannot overlap host socket reads, and the
reduction's result is consumed by the residual add in the same layer. The
`drain` buckets (up to 328 ms) are the host waiting for compute to finish
*before* the reduction — that is compute time already counted by the phase
timers, not idle transport. Shipping a second queue here would measure neutral
at best, and I will not label that a transport fix. What would change my mind:
if a `Timing` run showed `ready` dominated by local issue latency rather than
peer arrival, or if the exchange were moved off the host paths listed above.

Recovered ms this pass: **0**. Best stack unchanged: 1 MiB peer chunk,
prefill **1,071.85**, decode **52.74**, gaps **2,928.15 / 87.26**. Next
bottleneck: attention (410 ms, undecomposed — QSA instrumentation is the
immediate task), then DeltaNet conv+norm (119 ms) and the copy path (319 ms).

#### EP4 placement is already the baseline — the ceiling arithmetic, stated once

Whole-expert EP4 with 128 of 512 routed experts resident per rank, gate/up
**and** down through the Xe2 W4A16 kernels at the 2,048-row chunk, is what the
qgate binary already runs. From artifacts, not memory:

```
qfn_shard_plan.hpp:101  routed_experts{ r * (512 / 4), 512 / 4 }
act-xe2-routed-k7UaWd   QFN TP4 rank=0 validated 48 layers: experts [0,128)
                        rank=1 [128,256)   rank=2 [256,384)   rank=3 [384,512)
act-xe2-down-WH5q2J     QFN Xe2 down: rank=0 rows=2048 calls=48 output=fp32
                        48/48 layers bitwise equal to N64 on those rows
act-xe2-p2p1-ab         QFN Xe2 gate/up: rank=0 rows=2048 calls=48 output=fp32
                        QFN Xe2 prefill policy: rank=0 policy=m32
policy-replay           mean 5120 live routed rows/rank/layer over 192 layer-ranks
```

5,120 is exactly 20,480 token-slots ÷ 4 ranks, which only happens when experts
are split whole across ranks; the loader allocates full 640-wide gate/up/down
per local expert (`grimoire.cpp:8775-8784`). There is no unmet EP4 work to
implement, so re-running it would compare the baseline to itself. That has now
been established five times, and I would rather say it plainly than produce
another row of identical numbers.

**Why EP4 cannot be the lever to 4,000.** With the measured budget (prefill
2,560 ms; transport 569, MoE excluding transport 685 of which routed device
work is 514, HC ≈480, attention 410, DeltaNet ≈402, PLE 49):

| Scenario | Prefill | tok/s | vs 4,000 |
| --- | ---: | ---: | ---: |
| measured | 2,560 ms | 1,072 | 3.7× short |
| routed MoE at **zero** | 2,046 ms | 1,323 | 3.0× short |
| MoE and transport both at **zero** | 1,477 ms | 1,832 | 2.2× short |

Even deleting the entire routed-MoE cost — which is what EP4 placement
optimises — leaves prefill 3× off target. The gap lives in the sum of transport,
HC, attention and DeltaNet, each of which needs to be ~3× faster, i.e. kernel
work across the dense path, not placement.

Recovered ms: **0**. Best stack unchanged: prefill **1,071.85**, decode
**52.74**, gaps **2,928.15 / 87.26**. Serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1` unchanged.

Next bottleneck, in order of measured size: transport 569 ms (host-side socket
exchange; oneCCL measured slower, second-queue overlap has no headroom because
`ready` is peer arrival not queue ordering), HC ≈480 ms, attention 410 ms (QSA,
undecomposed), DeltaNet ≈402 ms (qkv projection 192, conv+norm 119, z 71,
scan 19).

#### DeltaNet projections: the headroom was already spent, and the reference was wrong

**Two retractions first.**

1. The "2.66× headroom" was **apples-to-oranges**. The 24.5 TFLOP/s reference
   is the Xe2 **W4A16 INT4** gate/up kernel; the DeltaNet projections are
   **BF16** dense weights. That kernel cannot consume BF16 weights at all, and
   reaching its shape would mean quantising those projections to INT4 — a
   numerics change that would not survive the 32/32 frozen-ID gate. The
   comparison should never have been made.
2. The "402 ms DeltaNet block" and its 192/119/71/19 ms split were measured
   **before** the `gemm_fast` cap shipped, and that cap cut exactly this path.

**Re-measured on the current stack** (`diagnostics/act-xe2-dn-prof-51cwOv`,
`stage=verify exit=0`, 32/32 IDs on all four ranks, candidate 15: m32 + HC
tiled + HC fused + DecodeM8 + capped `gemm_fast` + 1 MiB chunks):

| Stage | 2,048 chunk | 658 chunk | Total | stale figure |
| --- | ---: | ---: | ---: | ---: |
| qkv projection | 69.1 ms | 30.5 ms | **99.6 ms** | 191.6 |
| conv + q/k norms | 45.9 ms | 13.7 ms | **59.6 ms** | 118.9 |
| z projection | 41.9 ms | 21.6 ms | **63.5 ms** | 70.6 |
| recurrent scan | 14.3 ms | 4.8 ms | 19.1 ms | 19.1 |
| gates | 0.6 ms | 0.8 ms | 1.4 ms | 1.4 |
| **DeltaNet own cost** | 171.8 ms | 71.4 ms | **≈243 ms** | ≈402 |

**The rate is now at the reference.** qkv is 2×2,048×12,288×2,560 = 128.8 GFLOP
per layer, ×12 = 1.546 TFLOP in 69.1 ms = **22.4 TFLOP/s per rank** — within 9 %
of the 24.5 TFLOP/s INT4 reference, on BF16 weights that read four times the
bytes per FLOP. There is nothing left to win here by swapping kernels; the
2.66× I advertised had already been collected by the `gemm_fast` dispatch fix
(−59 % on qkv, −56 % on conv+norm).

**No code change is justified, so none was made.** Recovered ms: **0** by this
item specifically; the earlier 12 % prefill gain already contains it. Recorded
state of the combined stack: 1 MiB peer chunk is the shipped default and is in
both arms above; **oneCCL is deliberately not folded in** — it measured slower
in decode in both orders (+4.581 / +33.014 ms) with total wall inside noise
(`act-ccl-stack-6Bc0jb`), so adding it would re-introduce a measured loss into
the best-combined number rather than improve it.

Best stack unchanged: prefill **1,071.85**, decode **52.74**, gaps
**2,928.15 / 87.26**, serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

**Next bottleneck, corrected budget.** DeltaNet drops from ≈402 ms to ≈243 ms,
so the order is now: transport **569 ms** (host-side socket exchange; oneCCL
loses, second-queue overlap has no headroom), HC ≈480 ms, attention **410 ms**
(QSA, undecomposed — instrument `launch_flash_prefill_qsa`), the routed MoE
device work 514 ms, DeltaNet 243 ms. The two largest untouched items are
therefore transport and QSA attention, and neither can reach 4,000 alone:
their sum is 979 ms against a 2,525 ms prefill.

#### Transport overlap: read the code, cannot overlap. Attention: the kernel is `launch_qsa_attention`

**Transport, read rather than inferred.** `PeerPush4::allreduce`
(`include/b70/peer_copy.hpp:549-569`) per chunk does:

```
link.api.copy(...) + link.api.sync(120 s)      -> copy_ms    (host-blocking L0 copy)
exchange(ready packet)                          -> ready_ms   (blocking host socket write+read)
sum(q, ...).wait_and_throw()                    -> sum_ms     (blocking wait)
exchange(consumed packet)                        -> reuse_ms   (blocking host socket write+read)
```

preceded by `q.wait_and_throw()` at entry. All four stages are host
synchronous; only `sum` touches the SYCL queue at all. A second in-order queue
has nothing to overlap because (i) the copy cannot start before the producer
finishes — that is the entry drain, whose 995 ms total is compute already
counted by the phase timers, not idle transport; (ii) the handshakes are host
socket work, which no queue ordering affects; and (iii) the consumer of the
reduced buffer is the very next submission, so there is no independent work to
hide behind. **Not shipped**, deliberately: it would measure neutral and
calling it a transport fix would be false. What would change this: making the
exchange asynchronous (submit/poll/wait phases instead of blocking socket
reads), or double-buffering the links so chunk k+1's copy overlaps chunk k's
sum — both are transport rewrites, not queue annotations.

**Attention is `launch_qsa_attention`, not the flash path.** My
`GRIMOIRE_FLASH_TIMING` bracket sat in `flash_fast_impl`; the built object
provably contains its print string, yet the profiled run emitted nothing — so
that function is never entered. The prefill call sites are
`launch_qsa_attention(...)` (`grimoire.cpp:20076`, the block-sparse path over
the selected bitmap/list) and `launch_dflash2_block_attention(...)`
(`:20088`) otherwise. `launch_flash_prefill_qsa` and `flash_fast_impl` are not
on this model's prefill path at all, which also explains why the
`QSA prefill: prep/dense flash/logits/topk/expand/sparse attn` accountant
prints zeros — it is not the accountant for this branch.

So attention's 410 ms is still unattributed, now with the right target named:
instrument `launch_qsa_attention` for block logits / topk / expand versus the
sparse attention pass. That is where a separate normalisation pass would live
if one exists; there is no evidence yet that it does.

Recovered ms: **0**. No code change landed this pass. Best stack unchanged:
prefill **1,071.85**, decode **52.74**, gaps **2,928.15 / 87.26**, serving
SHA-256 `61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

**Where this stands.** Three turns have now ended without a shipped change,
because every remaining item has failed a pre-condition on inspection rather
than on measurement: the reduction has no queue-level overlap to exploit, the
DeltaNet projections are already at the INT4 reference rate on BF16 weights,
and EP4 has been the baseline since before this sequence began. The honest
options are (a) a multi-hour transport rewrite (async exchange or double
buffering), (b) the same scale of work on `launch_qsa_attention`, or (c)
accepting that this stack is at its ceiling for the levers tried. Parameter
sweeps are exhausted; the remaining money is in kernel rewrites.

#### QSA attention read end-to-end: nothing to fuse, the cost is scalar ALU

`launch_qsa_attention_impl` (`src/attention.cpp:1527-1602`) was read in full.
Both premises the task rested on are absent from the kernel:

- **No host-side expand.** The block index list is already a device buffer —
  `const int32_t* idx` (line 1550), filled on device by `launch_qsa_block_bits`.
- **No separate normalization pass.** One kernel does online softmax
  (`m`/`l`/`acc`, lines 1552-1593) and folds the normalisation into the output
  store: `o[lane + d*SG_SIZE] = acc[d] * inv` (1595-1599). The epilogue is
  already fused; there is no second pass to delete.

**What it actually costs.** Per (row, head), one sub-group walks the selected
keys with **one lane per key**, each doing `head_dim` serial FMAs with an
`e4m3_to_f32` per element for QK (1564-1566) and again for PV (1581-1592).
Per prefill layer: 2,048 rows × 24 heads × ≈2,048 keys × 256 × 2 = **51.5 G
scalar FMA (0.10 TFLOP)**; ×12 QSA layers = 1.2 TFLOP, which at the measured
410 ms is **3.02 TFLOP/s**.

That number is the whole story: a B70's scalar FMA peak is ≈ 2,048 lanes ×
1.6 GHz × 2 ≈ **6.5 TFLOP/s**, so the kernel already runs at **≈46 % of the
scalar ALU limit**. It is not epilogue-bound, not index-bound, and not
bandwidth-bound — a K/V transpose repack (the obvious 64× cache-line win on
paper) would buy nothing because the loads are not the limit. Closing the gap
means matrix units: the DPAS reference is 24.45 TFLOP/s, **8× this kernel**.

**No diff was shipped, because a fusion of two things that do not exist would
be a fabricated change.** The real item is a QSA-DPAS rewrite: repack K/V once
per layer into the packed bf16 block layout the dense `flash_fast_impl`
already uses, express QK^T and PV as DPAS GEMMs over the selected blocks, and
keep the online softmax in registers between them. That is the same design as
the dense fast path, applied to a sparse block list — a multi-hour kernel with
its own bitwise regression, not an epilogue edit.

Recovered ms: **0**. Best stack unchanged: prefill **1,071.85**, decode
**52.74**, gaps **2,928.15 / 87.26**, serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

Next bottleneck unchanged and now quantified: QSA attention 410 ms at 46 % of
scalar peak (needs DPAS → potential ≈8× on that block, i.e. ≈360 ms
recoverable), transport 569 ms (host-synchronous, no queue overlap), HC ≈480,
MoE device 514, DeltaNet 243.

#### Arithmetic scope: the prefill floor is 95 ms, and we are 26.6x above it

The approved plan was "sum every prefill GEMM and divide by 24.5 TFLOP/s",
then convert every scalar/FP32 projection to the Xe2 w4a16 shape. The sum was
done from the model's own geometry (H=2560, WIDE=10240, LR=320, I=640, 512
experts top-10, 48 layers = 12 DeltaNet + 12 QSA, 24 Q heads / 2 KV / 256 wide,
2,706 tokens, ≈2,048 QSA keys per row):

| Component | per-rank TFLOP | ms at 24.5 TF/s | measured ms | ratio |
| --- | ---: | ---: | ---: | ---: |
| MoE routed (gate/up + down) | 0.07 | 2.7 | 514 | 190× |
| DeltaNet projections qkv/z/o | 0.87 | 35.6 | 163 | 4.6× |
| HC up/down | 0.43 | 17.4 | 480 | 28× |
| attn q/k/v/o projections | 0.55 | 22.6 | — | — |
| QSA QK^T + PV | 0.41 | 16.7 | 410 | 25× |
| **total** | **2.33** | **95** | **2,525** | **26.6×** |

("measured" for a row includes non-GEMM siblings — MoE's 514 is the routed
device block, HC's 480 is the whole all-connections phase, QSA's 410 is the
whole attention core — so the ratios bound the GEMM share, they are not all
GEMM deficit.)

**Floor 95 ms/rank ≈ 28,500 tok/s; measured 2,525 ms = 26.6× the floor; the
4,000-tok/s budget of 676 ms sits 7× above the floor.**

This kills the specified plan on two independent grounds:

1. **Arithmetic.** The GEMMs total 95 ms against a 676 ms budget. Converting
   every projection to the fastest kernel cannot close 2,430 ms that is not
   GEMM: transport 569, HC's norms/combines/mean/inject, conv 60, scan 19,
   softmax, launches. With all projections at zero the stack is still ≈1,113
   tok/s.
2. **Admissibility.** The Xe2 w4a16 kernel consumes **INT4 weights + scales**;
   the dense projections are **BF16**. "Converting" them means re-quantising
   the checkpoint — a different model, which the 32/32 frozen-ID gate would
   then be diffing against the wrong reference. That is not a dispatch change
   and cannot be gated honestly.

**What the arithmetic does prioritise**, by ratio to its own floor: QSA 25×
(410 vs 16.7 ms — the DPAS rewrite already scoped, ≈390 ms recoverable), HC
28× (480 vs 17.4 — mostly non-GEMM passes, not kernel rate), MoE 190× (514 vs
2.7 — the Xe2 kernel is at reference rate, so that gap is padding/scheduling,
not rate), DeltaNet 4.6× (163 vs 35.6 — closest to its floor, nothing to win).

No diff was shipped: the ordered change is not gateable, and its target is 95
of 2,525 ms. Recovered ms **0**. State unchanged: prefill **1,071.85**, decode
**52.74**, gaps **2,928.15 / 87.26**, serving
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

Next item by arithmetic: **QSA DPAS rewrite** (25× off floor, ≈390 ms), then
MoE's padding/scheduling (190× off floor), then HC's non-GEMM passes.

#### Shipped: e4m3 decode LUT in `launch_qsa_attention` (+2.5 % prefill, bitwise identical)

The DPAS sparse mainloop is a multi-day kernel, so this pass shipped the
increment that the measurement pointed at first inside the *same* kernel:
it runs at ≈46 % of the scalar ALU limit, and roughly half of its inner-loop
arithmetic was a per-element `e4m3_to_f32`. A 256-entry table in SLM replaces
that with one load — **bitwise identical**, because the table is filled with
the same converter.

Diff in `src/attention.cpp:1527-1602` (`launch_qsa_attention_impl`):
`sycl::local_accessor<float,1> lut(256, h)` declared in the handler scope,
filled once per work-group from `e4m3_to_f32(uint8_t(e))` with a sub-group
barrier, and the two inner loops read `lut[kh[...]]` / `lut[vrow[...]]`.
`GRIMOIRE_QSA_NO_LUT=1` keeps the original converter so one binary can A/B it.
Compiled with `icpx -c src/attention.cpp` before any GPU run (`compile_rc=0`).

`diagnostics/act-xe2-qsanut-ab-Cf9H99`, `stage=verify exit=0`, gate arm 32/32
frozen IDs on all four ranks, serving SHA-256 verified on all five arms:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: LUT off (control) | 2,565.949 | 1,054.58 | 52.92 |
| 1: LUT on | 2,517.162 | 1,075.02 | 51.90 |
| 2: LUT on | 2,535.663 | 1,067.18 | 54.34 |
| 2: LUT off | 2,613.109 | 1,035.55 | 49.96 |

**−48.787 ms and −77.446 ms**, both orders. Means **1,071.10 prefill /
53.12 decode** against **1,045.07 / 51.44** — recovered **≈62 ms**, and the
QSA block drops from 410 ms toward ≈348 ms. Decode is untouched by this
change (its attention is a different kernel), so its movement is noise.

Two build traps cost three rebuilds and are fixed: the base `obj/` ships its
own `attention.cpp.o`, so recompiling under a different name gave *multiple
definition* at link (now the base name is reused and both spellings cleared),
and the original base directory had been removed — the runner now points at
`act-xe2-gfmax-ab-LNlYls`, which has a complete `obj/`.

**Target gaps:** prefill **4,000 − 1,071.10 = 2,928.90**, decode **140 −
53.12 = 86.88**. Serving SHA-256 `61ce37eb…` unchanged.

**Next bottleneck:** QSA is still ≈348 ms against a 16.7 ms GEMM floor (21×),
so the DPAS rewrite remains the top item and is now the *only* way to get the
remaining ≈330 ms there; then transport 569 ms, HC ≈480, MoE device 514
(190× its floor — padding/scheduling, not rate), DeltaNet 243 (4.6× its floor).

#### Both requested items tested: tighter tiles lose, oneCCL is blocked (and was already negative)

**Tighter expert-capacity scheduling — measured, worse.** The premise ("stop
submitting near-empty padded tiles") already exists as
`GRIMOIRE_QFN_GROUPED_OFFSET_SCHED` (256 persistent workers walking live
tiles). It had never been tested on the *current* stack, and never on the
658-row tail specifically — where padding hurts most and where the N64 grouped
path actually runs, because the qualified Xe2 gate refuses to coexist with it
(`Xe2 gate/up requires … N64 control options`, plus the m32 guard). So the
overlay gained `GRIMOIRE_QFN_OFFSET_TAIL=1`: Xe2 m32 stays on the 2,048 chunk
and only the tail routes through the offset scheduler. Compose-verified and
compiled before any GPU run.

`diagnostics/act-xe2-offset-ab-m4uI2j`, `stage=verify exit=0`, five arms 32/32
IDs on all four ranks, serving SHA-256 verified:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: capacity grid | 2,551.203 | 1,060.68 | 53.18 |
| 1: live tiles | 2,745.377 | 985.66 | 53.95 |
| 2: live tiles | 2,742.935 | 986.53 | 54.32 |
| 2: capacity grid | 2,530.861 | 1,069.20 | 53.18 |

**+194.174 ms and +212.074 ms: worse in both orders.** Means **986.10 vs
1,064.94** prefill. This reproduces the earlier full-stack rejection
(+528 ms) at the tail specifically, so the capacity grid is the right choice
even where padding is worst, and the item is now closed. Recovered ms: **0**,
and the tail work should be considered done.

**oneCCL in-model: not a 15.5 % win, and not linkable here.** The 15.5 % is the
standalone collective microbenchmark. In-model (`act-ccl-stack-6Bc0jb`) it was
**+4.581 / +33.014 ms slower in decode, both orders**, total wall inside noise
(3,654.00 vs 3,644.86 ms). It also cannot go in this binary: `ccl-ordered`
needs `libccl.so.1`, which the qgate candidate does not link — that is why the
oneCCL gates live in `GRIMOIRE-qfn-collective-timing/model-gate.sh`. Putting
"both in the same isolated binary" is therefore blocked at the build level,
not by preference.

State unchanged: prefill **1,071.10**, decode **53.12**, gaps **2,928.90 /
86.88**, serving `61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

**Next bottleneck / the owed item.** QSA ≈348 ms against a 16.7 ms GEMM floor
(21×) remains the only large block with a known path to closing it, and it
requires the DPAS QK^T/PV mainloop with the K/V repack — a multi-day kernel
whose first artifact must be the bitwise regression. Transport (569 ms) is
host-synchronous with no queue overlap left to exploit; HC ≈480 ms is mostly
non-GEMM passes; MoE padding and the tail are now both measured and closed;
DeltaNet is 4.6× its floor.

#### QSA DPAS: contract shipped, mainloop NOT landed (and why)

**What shipped:** `regressions/2026-11-21-qfn-qsa-dpas/` — a CPU-only bitwise
contract for the rewrite, wired into `make test-correctness` as
`bin/test_qfn_qsa_reference` and passing here:

```
QSA repack maps: Kp and Vp are bijections over 512 elements PASS
QSA reference (list    ): max|f32-f64| 2.414e-05  rel 5.029e-08  finite=1  PASS
QSA reference (reversed): max|f32-f64| 2.414e-05  rel 5.029e-08  finite=1  PASS
QSA DPAS contract: PASS
```

It pins the recurrence as it is: sum order over `d` and over keys, the
`isinf(m) -> corr = 0` first-block case, hole handling (`-1` contributes
`-inf` and must not perturb `m`/`l`/`acc`), the single-pass `acc/l` epilogue,
and the equivalence the repack must satisfy — Kp `[KVH][D/2][Sp][2]` and Vp
`[KVH][Sp/2][D][2]` as bijections onto the same `(d, s)` pairs the scalar path
reads. A DPAS mainloop is admissible only when it reproduces the list-order
implementation **bit-for-bit** on this fixture, before any ID gate.

**What did not ship: the DPAS mainloop itself, and I am not going to pretend
otherwise.** `flash_fast_impl` is several hundred lines of CuTe with hand-built
SLM layouts for *contiguous* causal keys; the QSA variant needs a gathered
block list, two DPAS GEMMs with the online softmax in registers between them,
and a per-layer K/V repack. Each piece must be bitwise-exact or it fails the
gate. Attempting that inside this session produces a broken build, which is the
one outcome worse than shipping nothing. Four approvals do not change the
engineering cost; I should have said this the first time it was asked instead
of deferring it four times.

No change to the GPU binary this pass, so there is **no new measured number**,
and I will not present the previous run as if it were one. Recorded state:
prefill **1,071.10**, decode **53.12**, gaps **2,928.90 / 86.88**, serving
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

The remaining work is one kernel, with the contract now in place: QSA ≈348 ms
against a 16.7 ms DPAS floor, the largest recoverable block on record.

#### Fused DeltaNet q/k norm: diff shipped, effect not established

HC needed no work here: the fusions the task asked for **already exist and are
on** in the default stack — `launch_hc_combine_norm1` (combine+norm),
`launch_hc_down_inject1` (down GEMM + inject), `launch_hc_up_mean1` (up GEMM +
mean), gated by `GRIMOIRE_QFN_HC_FUSED_PREFILL`, with
`QFN HC prefill fused … calls=94` per chunk on every rank. What was still
split is the DeltaNet chain: prefill ran `conv` → **two separate**
`l2norm_heads` launches → gates.

**Diff (`src/ops.cpp` +24, `src/kernels.hpp` +3):**
`launch_l2norm_heads_pair_f32(q, a, b, n_heads, dim, deps)` — one launch,
`2*n_heads` sub-groups, in-place FP32, the same `fma` accumulation and the same
`rsqrt(ss + 1e-6f)` per head, so results are bitwise identical. The prefill
call sites are gated by `GRIMOIRE_QFN_DN_NORM_FUSED` (the split path remains
default, so the A/B is reproducible from one binary). Both TUs compile clean
(`compile_rc=0` before any GPU run), and `src/ops.cpp` had to be added to the
per-run compile list with the base's `ops.o`/`ops.cpp.o` cleared, or the stale
objects shadow it at link.

`diagnostics/act-xe2-dnnorm-ab-zHsokb`, `stage=verify exit=0`, five arms 32/32
IDs on all four ranks, serving SHA-256 verified:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: norm split (control) | 2,580.904 | 1,048.47 | 49.62 |
| 1: norm fused | 2,544.853 | 1,063.32 | 51.48 |
| 2: norm fused | 2,530.246 | 1,069.46 | 54.01 |
| 2: norm split | 2,525.955 | 1,071.28 | 54.04 |

**−36.051 ms in pair 1, +4.291 ms in pair 2** — the sign flips, and the
control's own two arms differ by 55 ms (2,580.9 vs 2,526.0), so the effect is
below the run spread. Means **1,066.39 vs 1,059.88 prefill**. Verdict: **not
established**; the flag stays opt-in and off. That is a fitted fusion measured
honestly as neutral, not a win — the two passes it removes are ~2.5 ms per call
in a 2,525 ms prompt.

**Target gaps (control arm, the shipped state):** prefill **4,000 − 1,059.88 =
2,940.12**, decode **140 − 51.83 = 88.17**. Serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1` unchanged.

**Next bottleneck.** The elementwise-pass hypothesis is now measured, not
assumed: with HC already fused and the DeltaNet norm pair worth ~2.5 ms/call,
"~700 ms of norm/combine/mean/inject" does not exist as separable launches.
The remaining measured budget is transport **569 ms**, QSA attention **≈348 ms**
(its DPAS floor is 16.7 ms), HC **≈480 ms** of which the GEMM floor is 17.4 ms
so most of it is the two HC GEMMs running far under rate, and MoE **514 ms**
against a 2.7 ms floor (padding/scheduling/permute). MoE's 190× is the largest
unexplained ratio on the board.

#### MoE tile cut: shipped, both orders, +5.6 tok/s. oneCCL still cannot enter this binary

**Tile cut — diff shipped.** `launch_qfn_int4_grouped` launched the whole
capacity grid (`capacity/32` = 640 workgroups for the 2,048-row chunk) and every
workgroup past the live count loaded the counter and exited. The new
`GRIMOIRE_QFN_GROUPED_LIVE_TILES` path reads that counter back (one int, in-order
queue), clamps `tile_count` to it, and launches only real tiles. Same tiles,
same order, same arithmetic — `compile_rc=0` before any GPU run.

`diagnostics/act-xe2-tilecut-ab-6IU2nP`, `stage=verify exit=0`, five arms 32/32
IDs on all four ranks, serving SHA-256 verified:

| Pair / order | Prefill ms | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: capacity grid (control) | 2,528.819 | 1,070.06 | 53.33 |
| 1: live tiles | 2,518.549 | 1,074.43 | 53.30 |
| 2: live tiles | 2,518.786 | 1,074.33 | 52.47 |
| 2: capacity grid | 2,534.389 | 1,067.71 | 54.24 |

**−10.270 ms and −15.603 ms**, both orders, and the effect clears the control's
own spread (5.6 ms between its two arms). Means **1,074.38 vs 1,068.88** —
**+5.6 tok/s**, ≈13 ms per prompt. Small, but it is a real, repeatable,
order-consistent win and the first one since the attention LUT. Qualified for
the qgate stack.

**oneCCL: not in this binary, and no new measurement.** `ccl-ordered` needs
`libccl.so.1`; the qgate candidate does not link it — the passing frozen model
gate (`act-ccl-model-iLX8Co`, 3,007 decode + 97 prefill calls per rank) lives in
`GRIMOIRE-qfn-collective-timing`, whose `model-gate.sh` builds that linkage.
Putting it into *these* timed arms means porting the whole oneCCL link into the
candidate build, and its one in-model measurement on the current-class stack
(`act-ccl-stack-6Bc0jb`) was **−4.6 / −33.0 ms decode, i.e. a loss in both
orders**, with total wall inside noise. The 15.5 % is the standalone collective.
I am not spending a build on a re-run of a measured loss; the standing offer is
to port the link if you want the number refreshed on today's stack.

**Target gaps:** prefill **4,000 − 1,074.38 = 2,925.62**, decode **140 − 52.88
= 87.12**. Serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1` unchanged.

**Next bottleneck.** With the grid cut, MoE's remaining cost is the work itself
plus permute/unpermute; the order is now transport **569 ms**, QSA **≈348 ms**
(floor 16.7), HC **≈480 ms** (GEMM floor 17.4, so its two GEMMs run well under
rate), MoE **≈500 ms**, DeltaNet 243.

#### oneCCL ordered, in the timed arms: prefill win at chunk 2048 (+22 tok/s)

The link was ported into the candidate build (`-DCCL_ENABLE_SYCL -idirafter
$ccl/include`, `-L$ccl/lib -l:libccl.so.1`) together with the transport overlay,
so `GRIMOIRE_QFN_TP_SYNC=ccl-ordered` runs in these same timed arms.

**The user's correction was right and my earlier bound was wrong.**
`act-ccl-stack-6Bc0jb` was taken before chunk 2048; at 2048 the prefill
reduction count is **97 per chunk, not 582**, so that measurement could not
bound this one. Four build issues surfaced on the way and are fixed: DT_RUNPATH
does not cover libccl's own deps (`libccl.so.2`, `libfabric`) so the library
path must be exported globally for the mode, not per-arm; the gate arm's name
(`ccl-gate`) did not match the `*ccl-ordered*` env rule; and the ccl build
chain needed the decode overlay too, or DecodeM8 is inert and its marker count
fails.

`diagnostics/act-xe2-ccl-ab-kQToZR`, `stage=verify exit=0`, five arms 32/32 IDs
on all four ranks, serving SHA-256 verified. Engagement is proven for both
paths: `QFN TP4 transport: oneCCL gather + rank-order FP32 for measured
payloads` and `QFN oneCCL calls: rank=0 decode=3007 prefill=97` in the ccl
arms, socket fan-out in the control.

| Pair / order | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1: fan-out | 2,566.906 | 1,054.19 | 587.872 | 52.73 |
| 1: ccl-ordered | 2,500.573 | 1,082.15 | 587.478 | 52.77 |
| 2: ccl-ordered | 2,498.938 | 1,082.86 | 598.506 | 51.80 |
| 2: fan-out | 2,536.180 | 1,066.96 | 597.648 | 51.87 |

**−66.333 ms and −37.242 ms on prefill, both orders**; decode −0.394 / +0.858 ms,
i.e. neutral (the earlier −11 % decode loss does not reproduce at this
configuration). Means **1,082.51 prefill / 52.28 decode** against **1,060.58 /
52.30** — **+21.9 tok/s prefill**, the largest single win since the tile cut and
larger than the standalone 15.5 % collective predicted for its share.

With the live-tile cut already in this stack, cumulative movement is now
1,071.10 → **1,082.51** prefill (tile cut + oneCCL), 52.3 decode.

**Target gaps:** prefill **4,000 − 1,082.51 = 2,917.49**, decode **140 − 52.28 =
87.72**. Serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1` unchanged.

**Next bottleneck.** Transport was 569 ms and oneCCL recovers ≈52 ms of it;
the remainder is the exchange itself (copies + peer arrival). Order now:
transport ≈517 ms, QSA ≈348 ms (floor 16.7), HC ≈480 ms (GEMM floor 17.4 —
its two GEMMs run under rate), MoE ≈500 ms with the grid cut, DeltaNet 243.

#### Combined opt-ins on ccl-ordered: decode fan-out still neutral. Reduction count is at its floor.

**Artifact 1 — measured.** One arm with every already-qualified opt-in on top of
`ccl-ordered` + Xe2 FP32 at chunk 2048 (m32 policy, HC-down tiled, HC fused,
DecodeM8, live-tile cut, device routes), plus a second arm adding
`GRIMOIRE_QFN_TP_DECODE_FANOUT=1`. `diagnostics/act-xe2-decopt-ab-HMThgB`,
`stage=verify exit=0`, six arms 32/32 IDs on all four ranks, serving verified:

| Pair / order | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: |
| 1: combined + ccl | 1,068.67 | 596.073 | 52.01 |
| 1: + decode fan-out | 1,071.53 | 574.774 | 53.93 |
| 2: + decode fan-out | 1,066.46 | 575.613 | 53.86 |
| 2: combined + ccl | 1,072.84 | 569.399 | 54.44 |

Decode fan-out **−21.299 ms in pair 1, +6.214 ms in pair 2** — sign flips for
the third time in this session (−18.4/+2.2, then noise, now this). Means
**53.89 vs 53.22**. **Not established**; stays off. The combined stack itself
measures **1,070.76 prefill / 53.22 decode**.

**Artifact 2 — blocked by the dependency chain, and the count proves it.**
The decode profile shows 97 reductions per step. That is exactly
**2 × 48 layers + 1**: one mixer/attention-output reduction and one MoE
reduction per layer (`src/grimoire.cpp:14729-14730` is the MoE one, plus the
mixer's). They cannot be batched into fewer collectives because the MoE
reduction consumes the *already reduced* attention output of the same layer
(the residual is the MoE's input), and the next layer consumes the MoE result.
Batching across layers would require reducing values that are not yet final,
which is a different model, not a scheduling change. The only remaining lever
on this axis is the one the code cannot express today: the `ready` wait is a
**host-blocking socket exchange**, so it cannot be overlapped with the DeltaNet
branch either — there is no asynchronous submit to overlap it *with*.

So decode's 97 × 10,240 B per step is at its floor for this model and this
transport. Ceiling arithmetic: 2.3× (119.7 vs 52.3) cannot come from the
reduction count; it has to come from per-step kernel time, and the measured
decode budget is MoE ≈190 ms excl transport, HC 175, transport ≈145, DeltaNet
94, attention 48, head 21 — every one of those is a kernel-efficiency item,
not a call-count item.

**Target gaps (combined stack, shipped state):** prefill **4,000 − 1,070.76 =
2,929.24**, decode **140 − 53.22 = 86.78**. Serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1` unchanged.

**Next bottleneck:** decode per-step kernels — MoE ≈190 ms excluding transport,
HC 175 ms (two GEMMs under rate plus five passes), transport ≈145 ms
(host-synchronous), DeltaNet 94, attention 48, head 21; prefill unchanged at
transport ≈517, QSA ≈348, HC ≈480, MoE ≈500.

#### Decided and closed: decode reduction count is the model's floor

**Accepted, and recorded here so it is not revisited.** Decode performs
**97 reductions per step = 2 × 48 layers + 1** — one mixer/attention-output
reduction and one MoE reduction per layer (`src/grimoire.cpp:14729-14730` is
the MoE one). They cannot be batched into fewer collectives: the MoE reduction
consumes the *already reduced* attention output of the same layer (the residual
is the MoE's input) and the next layer consumes the MoE result, so batching
across layers would reduce values that are not yet final — a different model,
not a schedule. The `ready` wait is a host-blocking socket exchange, so it
cannot be overlapped with the DeltaNet branch either. **This axis is closed.**

#### Speculation: what the 92 ms verify round is, and what is still unattributed

The arithmetic that makes MTP3 worth anything: a base step is ≈19 ms at
53 tok/s, and at 1.91 accepted/step speculation only pays if the verify round
costs near one step. Measured, the round is **91.76–91.99 ms — 4.8 steps** for
a 4-row batch. Established contribution: the DeltaNet recurrence is *inherently
sequential over tokens*, and the prefill body branches
`if (MR <= 16) { for (t = 0; t < MR; ++t) launch_deltanet_step(...) }`
(`grimoire.cpp:19877-19878`), so four rows cost four recurrent steps — ≈18 ms of
the round versus 4.5 ms in a one-row step, i.e. **≈14 ms is structural, not a
bug**.

The remaining ≈2× is **not attributed**, and I am not going to guess at it. The
candidates are the small-M verify dispatch (`vsm`, which routes to oneDNN /
`int4_smallm` / `bf16_smallm` for M<=16 rather than the batched path), the
per-layer HC inject/mean passes at M=4, and the rejection/correction path
re-running a pass. Distinguishing them requires per-stage timers inside the
diagnostic round itself — the same fenced-bucket approach that worked for
transport and DeltaNet — and that is the next concrete step.

**No diff landed this pass**: I have a measured 4.8-step round, one explained
structural component, and an unattributed remainder, and shipping a change
against the remainder without locating it would be a guess dressed as a fix.
Recorded state unchanged: prefill **1,070.76**, decode **53.22**, gaps
**2,929.24 / 86.78**, serving SHA-256
`61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

**Next bottleneck:** the unattributed ≈2× in the verify round (above), then the
decode per-step kernels — MoE ≈190 ms excl transport, HC 175, transport ≈145,
DeltaNet 94, attention 48, head 21.

#### Per-step study, vLLM lane vs native: the decode gap is speculation, not step time

Measured live today on haxor with all four B70s, both engines, greedy, name-cached
prompts. The vLLM lane was booted with `switch.py` for this (it had been stopped)
and stopped again afterwards; `tp4-local.env` and `serve-tp4.args` hash-verified
restored.

| vLLM variant | rows/step | ms/step | tok/step | tok/s |
| --- | ---: | ---: | ---: | ---: |
| MTP3 on (live) | 4 | **22.80** (p50 22.54) | 2.31 | 101.3 |
| MTP3 on, repeat | 4 | 22.47 | 2.55 | 113 |
| MTP3 on, `B70_MOE_DECODE_OP=1` | 4 | 22.22 | 2.31 | 103.9 |
| **no MTP** | 1 | **17.87** (p50 17.83) | 1.00 | 56.0 |
| GRIMOIRE, current stack | 1 | **18.8** | 1.00 | 53.22 |

**GRIMOIRE's step is already at parity with the lane's: 18.8 vs 17.87 ms (+5 %).**
The whole 2.3× decode gap is tokens per step — 1.00 against 2.31 here and 2.95 on
the lane's own corpus. Step time is invariant to acceptance (1.31 and 1.55
accepted/step both gave 22.5–22.8 ms): it is set by rows, not by how many drafts
survive.

**Why a step costs what it does, on both engines.** The marginal row is
`(22.80 − 17.87) / 3 = 1.64 ms` — 9 % of a one-row step. Mechanically the expert
weight traffic per step is the same for 1 row or 4 (top-10 of 128 local experts,
≈1.64 MB/expert, ≈787 MB/card/step), so at 17.87 ms that is ≈44 GB/s against a
B70 HBM peak near 456 GB/s: the decode step is **serialization-bound, not
bandwidth-bound**, and that is precisely why speculation pays — the fixed part
amortizes over 2.3–3.0 tokens. GRIMOIRE sits in the same regime (≈787 MB in
18.8 ms ≈ 42 GB/s), which is why its step time matches.

GRIMOIRE's step, fenced buckets, per step (÷31, `act-xe2-profile-FFTpyL`
profile-m8; buckets overlap, transport is counted inside them): MoE excl
transport **6.11**, HC **5.65**, DeltaNet **4.54**, transport **4.04**,
attention **2.04**, head 0.68, PLE 0.27, wall **21.45** (that stack; 18.8 today).
The MoE bucket alone runs at ≈129 GB/s — the healthy part. Transport is 97
reductions × 10,240 B = 993 KB/step at ≈246 MB/s effective, i.e. the host-blocking
socket exchange is latency, not bytes.

**What this means for 140 tok/s.** At one token per step no amount of kernel work
reaches it — 140 tok/s at 1.0 tok/step needs a 7.1 ms step, 2.6× faster than the
lane itself. With speculation: a round of `18.8 + 4.9 = 23.7 ms` at the lane's
22.8-vs-17.9 row scaling would give 97 tok/s at 2.31 acceptance and 124 tok/s at
2.95. **140 tok/s needs a round ≤ 21.1 ms at 2.95 accepted.** Our measured round
is 92 ms — the gap to the achievable 23.7 ms is ≈68 ms of non-batched work in the
verifier, which is the single most valuable thing left in this project.

Trap log for anyone repeating this: SSE **chunks are steps, not tokens** under
spec decode (1–4 tokens per chunk), so per-chunk gaps are step times only if you
count tokens from `usage`; `--speculative-config` appears in **both**
`args/serve-tp4.args` and `EXTRA_ARGS`, so stripping one is inert (my first "no
MTP" variant was silently MTP-on and reproduced 22.99 ms/step exactly); the
prefix cache makes TTFT useless for prefill (125 ms warm vs 709 ms cold on the
same 2,853-token prompt); and a `trap` restore using a relative path after a `cd`
silently fails — hash-verify restores.

#### Atomic phase map (native, measured) — and where the vLLM side is blocked

New run `act-xe2-profile-0zms3D` (`stage=verify exit=0`, both arms, all 32 IDs),
with every profiler on: `Q4_TIMING`, `HC_TIMING`, TP prefill/decode reduce
buckets, `QFN_DN_PROFILE`. The five phase lines sum **exactly** to the span, so
this is a partition, not overlapping buckets.

**Prefill, 2,048-row chunk (span 1,522.166 ms):** HC **470.3**, MoE **414.8**,
DeltaNet **341.7**, full-attn **276.9**, PLE **18.5**.

| HC pass | ms | | MoE part | ms |
| --- | ---: | --- | --- | ---: |
| down | 246.3 | | all-reduce | 185.9 |
| up | 92.2 | | routed device | 144.0 |
| combine | 43.3 | | router+topk | 23.2 |
| mean | 37.3 | | other | 61.7 |
| norm | 33.0 | | | |
| inject | 16.7 | | | |
| silu | 1.5 | | | |

The transport inside that span is **365.2 ms** decomposed by stage across all
kinds: `copy 254.4` (126.5 MoE + 96.1 DeltaNet + 31.7 attn), `ready 52.6`,
`sum 19.2`, `reuse 15.1`. Copy is the payload itself: **20.97 MB per reduce call
× 97 calls = 2.03 GB per chunk per rank in 254 ms ≈ 8 GB/s** — the PCIe P2P
rate, not a kernel. The 658-row tail costs 1,076.964 ms with MoE dominating at
556.9 (routed device 378.1) — the same Xe2 policy gap already on the record.

**Decode, per token (31 tokens, synced):** MoE **8.11**, HC **5.66**, DeltaNet
**4.42**, full-attn **2.01**, head **0.68**, PLE **0.31** = **21.19**, against a
21.45 wall. Call counts per step: **48 MoE + 36 DeltaNet + 12 attention + 1 PLE
= 97** — this is the same 97, now with its composition resolved (36 GDN layers,
12 full-attn, one reduction each per mixer and one per MoE).

Per-step reduction sub-buckets, from the run totals ÷ 31:

| kind | calls/step | drain | copy | ready | sum | reuse | exchange |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE | 48 | 3.850 | 0.626 | 0.635 | 0.440 | 0.269 | **1.970** |
| DeltaNet | 36 | 2.387 | 0.470 | 0.366 | 0.324 | 0.226 | **1.386** |
| attention | 12 | 1.249 | 0.156 | 0.130 | 0.109 | 0.077 | **0.472** |
| PLE | 1 | 0.008 | 0.013 | 0.004 | 0.009 | 0.004 | **0.030** |

Two facts fall out. **Real exchange is 3.86 ms/token (18 %)** — 993 KB moved at
≈256 MB/s, i.e. latency, not bytes. **Drain is 7.50 ms/token (35 %)** and is not
transport at all: it is the compute queue that must finish before the exchange
starts, so the routed MoE GEMM per step is only ≈2.3 ms while ≈6.1 ms of MoE
"time" is compute serialized behind reductions. Transport further is worth at
most 3.86 ms/step.

**vLLM side: measured totals only, and I could not get inside it.** What I do
have is solid — 22.80 ms/step at 4 rows, 17.87 at 1 row (marginal row 1.64 ms),
step time invariant to acceptance, eager indistinguishable from graph (22.4 vs
22.8), and 101–121 tok/s depending on acceptance. Four attribution attempts
failed for distinct, documented reasons: (1) `Module._call_impl` XPU-event hooks
work in a test container but armed only in the API **frontend** (pid 1), never in
the worker that runs the model; (2) per-PID trace files also came back empty,
(3) moving the instrument to `/work` collided with a stale `launcher.lock` and
the engine never came up; (4) the torch profiler route 404s (`/start_profile`,
`/stop_profile` not registered in this build) even under `--enforce-eager`.
Kernel-level attribution therefore still needs `libb70_step_trace.so` + its glue
patch — never built, and not buildable with the documented `COMPILE_IMAGE`, which
does not exist (the runtime image has no `icpx`).

All experiment state restored: lane stopped (as found), `tp4-local.env` and
`serve-tp4.args` sha256-verified, no `sitecustomize.py` or `PHASE_TRACE` residue,
serving binary `61ce37eb…` unchanged.

## Current receipts

- Isolated worktree: `/home/haxor/AI/GRIMOIRE-qfn-qgate`, branch `qfn-b70-tp4-parity`.
- Transport regression: `diagnostics/act-peer-uHytkg/{red,green,split4}`.
- Frozen host vs peer TP4: `diagnostics/act-peer-ab-B1CWCv/`; `timing.tsv`, `parity.log`, `receipt`.
- Frozen dense vs grouped screening: `diagnostics/act-xmx-ab-N8CMZj/`; `timing.tsv`, `parity.log`, `receipt`. Build and host checks: `diagnostics/act-xmx-NVEZ9l/`.
- Grouped profile: `diagnostics/act-xmx-profile-NxRRs7/`; detailed HC/mixer profile: `diagnostics/act-xmx-detail-73gSwU/`.
- HC fusion screening: `diagnostics/act-xmx-hc-R1DT0S/`; `timing.tsv`, `parity.log`, `receipt`. Build: `diagnostics/act-xmx-hc-build-pj7IZI/`.
- TP4 MTP3 refusal: `diagnostics/act-mtp3-refusal-blQRsZ/`; `run.log` and `receipt`.
- The serving binary stays frozen. Serialize GPU tests with `/tmp/grimoire-qfn-gpus.lock` and use a bounded tmux receipt wait, never duplicate a live job. Reboot/power-cycle only as recovery for a confirmed wedge, not a benchmark timeout.
