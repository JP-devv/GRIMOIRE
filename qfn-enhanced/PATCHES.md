# The custom build — patch series, SYCL kernels, Level Zero shim

Everything that makes this vLLM not-vLLM. Base is upstream **vLLM v0.30.0 = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`**;
the fork is `Lumnus/vllm` branch `b70/v0.30.0-stable`, tag **`v0.30.0-b70.2`** (`9d79d28d7e32…`, 35 commits: 33
into `vllm/`, 3 into `tests/`).

## How the tree is built and verified

`patches/series.txt` maps fork-commit subject keys → patch filenames, in two columns (`vllm/` diff, `tests/`
diff). `scripts/export-series.sh` walks `git rev-list ced6857a..REF`, keys each commit by the first word of its
subject (colon stripped), and writes the two trees with `format-patch -1 --no-signature -- vllm/` / `-- tests/`
(appending, so several commits can share one tests file). It **aborts** on an unknown key, on a commit touching
files outside `vllm/`+`tests/`, or when a commit touches one tree but `series.txt` names `-` for it.
`scripts/check-series.sh` re-exports to a tmpdir and `diff -r`s both trees.
`scripts/make-tree.sh <release> <dir> [--from-patches]` rebuilds a tree onto a `ced6857a` checkout and diffs it
against the fork branch, then runs `image/verify-tree.py`, which sha256-checks every file the series touches
against the `PY_FILES` (wu1ff layer) and `B70_FILES` (final, overriding) blocks parsed out of
`image/verify-overlay.sh`. `UPSTREAM` in that file means "touched by the fork, then restored to upstream".

`Dockerfile.tp2` applies 0001–0032 by checking out the fork tag and **asserting the release's touched-file set
equals the patch set** (`git diff --name-only --no-renames BASE TAG -- vllm`), installs the files into both
`/opt/venv/lib/python3.12/site-packages` and `/workspace/vllm`, then applies **0040–0050** with
`patch -p1 --fuzz=0 --forward --dry-run` and then for real, in `LC_ALL=C` filename order, failing on any
`.orig`/`.rej`. Then `verify-tree.py` + `py_compile` of every touched file.

## The series, by theme

44 files in `patches/vllm/` (33 from the release tag + 11 unreleased, 0040–0050), 7 in `patches/vllm-tests/`
(never applied to the image), 8 in `patches/vllm-xpu-kernels/` (one superseded, 7 in the shipped `b70.3` wheel).
File paths below are relative to `vllm/`.

### Model support — 0001–0005, re-derived from wu1ff's MIT `B70-LLM-Controller`

| patch | what | files | env |
|---|---|---|---|
| 0001 | narrow the Qwen4Exp XPU `NotImplementedError` to TPU | `models/qwen4_exp/__init__.py` | — |
| 0002 | PLE table → two pinned slabs + UVA view, capture-stream fix | `qwen4_exp/nvidia/ngram_embedding.py` | `PLE_TABLE_PATH` |
| 0003 | HC down-GEMM deterministic K=5120 split | `nvidia/hyperconnection.py`, `model_executor/layers/linear.py` | — |
| 0004 | GDN speculative-decode metadata fusion (36 builders) | `v1/attention/backends/gdn_attn.py`, new `gdn_spec_metadata{,_batched}.py`, `v1/worker/gpu/attn_utils.py` | — |
| 0005 | route GDN to the int64-offset `libgdn_index64.so` | `_xpu_ops.py` | (lib **not installed** — see kernels) |

### PLE — the n-gram embedding table (0006–0008, 0013/0013b, 0050)

The model has a 320 001 536 × 160 n-gram embedding table. 95.4 GiB in BF16, which 61 GiB of host RAM cannot
pin. The series walks the table down: 0006 fill the pinned slabs straight from the mmap per TP rank (boot host-RAM
peak 191 → 158 GiB); 0007 keep it FP8 E4M3 with one global scale (95 → 48 GiB, a 256-entry dequant LUT, bitwise
equal to the stock cast); 0008 INT8 with one fp32 scale per row (164 B/row, 48.9 GiB over 4 ranks); 0013 stop
pinning it at all — serve it from NVMe through `O_DIRECT` with a CLOCK row cache and a host-side n-gram hash;
0013b add a native C reader (`native`, one call per batch with the GIL released) and an io_uring reader
(`uring`), plus prefill lookahead; **0050 read the FP8 shards in place from the checkpoint** (160 B/row, the
checkpoint's own values, no 52.5 GB INT8 copy) through the same 0013 row cache. **0050 is what runs here.**
Boot cross-checks 65 536 rows byte-for-byte against the mmap, and refuses to start above a relative L2 error
(0.25 FP8 / 0.02 INT8) on 4 096 random rows.

### Chat API behaviour (0009, 0010, 0012)

Per-effort thinking budgets keyed on the client's requested effort, a server default `presence_penalty`,
`"ultra"` as an accepted effort, vLLM's built-in n-gram repetition stop as a server default (this is what catches
the token-1023 "duct" NaN loop), and an OpenRouter `{"reasoning":{"effort":X}}` alias.

### KV offload (0014a–f, 0018, 0031) — off on this lane

0014a trace; 0014b the **hybrid junction heal** — when a Mamba/GDN group cuts an offload hit below what the
full-attention groups proved, pin a shared-prefix junction there, so a "same document, new question" revisit
misses once then hits; 0014c retain GDN states at N block boundaries below the replay boundary so the *first*
revisit hits too (cost N × ~116 MB per request); 0014d a guard for vllm#56795; 0014e the upstream #51787
backport (request-scoped recency, tail-before-head eviction across all KV groups) in new `*_group_evict.py`
files with `policies/lru.py`/`arc.py` restored to upstream byte-for-byte; 0014f gates for 0014e. **0018** chunks
the pinned CPU KV pool — one allocation ≥ ~31 GiB per rank segfaults under torch 2.13 (the workaround was
`NEOReadDebugKeys=1 AllowUnrestrictedSize=1`). **0031** DMA CPU→GPU KV loads straight from the pinned pool —
leave it off with kernels b70.3, which have B70-K1.

Only 0014b (`B70_OFFLOAD_JUNCTION=1`) and 0014c (`_GDN_BACKSTEP=1`) are on here, and there is **no host-RAM KV
tier** (`kv_offloading_size=None`), so those two are inert until one is added.

### MTP / GDN correctness (0021, 0022, 0026, 0023/0023c+0024t)

0021 keeps the MTP draft on FULL (uniform-decode) CUDA graphs only on XPU — PIECEWISE draft-prefill capture
failed there; mixed prefill batches run the draft eagerly. **On here.** 0022 is a port of vllm#55506 (Mamba spec
block tables by request slot), inert at PP=1. 0026 never schedule a partial draft list (grammar-trimmed drafts
crashed the stock 0.1.14 GDN kernel; fixed upstream by kernels #600, so it is belt-and-braces with b70.3).
0023/0023c+0024t are debug: a GDN op-mode switch, a NaN tracer at the GDN wrapper and the sampling boundary, and
a probe that zeroes block 0 around FULL replays — driven by fixed flag files under `/run/b70-probes/` re-read
every 2 s, so they can be toggled in a live engine.

### MoE hot/cold experts (0040–0049) — the two-card machinery, all OFF here

For the 2× B70 case where VRAM cannot hold all 512 routed experts: 0040 split hot (VRAM) / cold (pinned host RAM
behind UVA), exact — hot + cold equals all-in-VRAM bit for bit, MTP layers never split; 0041 count routes per
(layer, global expert); 0042 prefill-time cold prefetch into two VRAM slots with layer L+1 DMA'd on a copy
stream while L computes (2-card prefill ~2 600 → ~3 950 tok/s); 0043 traffic-driven expert swapping in place, no
graph recapture; 0044 move `embed_tokens` into pinned host RAM behind UVA (0.59 GiB/card at TP2) and spend the
VRAM on hot experts, plus swap-prior tuning; 0046 route trace for cache studies; 0047 decode-aware ranking with
a learned hot set that survives restarts; 0048 the fused SYCL decode op + an FP8 draft head; 0049 hand a
percentage of each layer's cold experts to a host AVX-512 BF16 pool behind a host-memory mailbox.

**All off on this lane** — four cards hold every expert (EP4 = 128/card), and 0049 additionally needs AVX-512
BF16, which the i7-13700K does not have.

### 0045 dense FP8 W8A16 — off

A regex selects 16-bit linear layers of an INC checkpoint and quantises them at load to FP8 E4M3 (one scale per
output channel), run by the XPU weight-only kernel `fp8_gemm_w8a16`; `lm_head` never. Measured on 2× B70: MMLU
87.33 → 87.23 % (3 000 paired questions, p = 0.79), HumanEval 157 → 158/164, perplexity +0.4–0.7 %, for −0.9
GiB/card. Off here because VRAM is not the constraint on four cards and the four-card release ran BF16.

### 0029 / 0029b — one of these is dead

Full-vocab prompt logprobs spill ~3 GiB per rank into host RAM and are never returned. **0029 patches the V1
runner (`v1/worker/gpu_model_runner.py`), which this engine never calls — it does nothing.** 0029b patches Model
Runner V2 (`v1/worker/gpu/sample/prompt_logprob.py`) and is the one that acts. Both ship.

### Superseded, folded, and gaps

- **0029 is inert** (above). **0014e is only safe behind 0014f**, which restores `lru.py`/`arc.py` to upstream and
  moves the backport to new files; 0014e is still required for those new files.
- **0013b** adds the readers/lookahead on top of 0013 and defaults to 0013's v1 behaviour.
- **0032 supersedes 0027's** warning behaviour (both touch `_xpu_ops.py`; 0023 touches it too, its hash
  overwritten). 0027 makes `libgdn_index64.so` optional, 0032 warns once — with b70.3's kernels the stock GDN op
  is 64-bit, so the library is not installed at all.
- **0030** was the AWQ-branch duplicate of 0029b, consolidated away in b70.2. **0011** exists only as
  `0011-…patch.draft`, never published. **0024t** is folded into the 0023c file.
- **0015–0017 and 0033–0039 have no trace anywhere** — no patch, no doc, no git log. Unexplained gaps.
- **0051 exists but is not in the series** — see the kernels section.

## The SYCL kernels

Three packages under `kernels/`, all `TORCH_LIBRARY` torch extensions loaded with
`torch.ops.load_library(<path>)` — no ctypes, no raw Level Zero module load. Compiled AOT-only for Battlemage:

```sh
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" \
  -O3 -std=c++17 -shared -fPIC -ffast-math -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include -x c++ <src>.sycl -o <out>.so \
  -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib
```

`$T` is the runtime's own torch dir, so the `.so` is ABI-locked to `torch 2.13.0+xpu`. `-ffast-math` is
moe-decode only; `b70-step-trace` uses `-O2` inlined in its `build.sh` and has no `compile.sh`. The shipped `.so`
carries a single `__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen` bundle — **no JIT fallback**, so it will not load on
a different GPU generation.

**`b70-moe-decode`** (913 lines; patches 0048/0049/0051) — decode-size (≤16 tokens, ≤256 route slots) INT4 W4A16
MoE doing hot and cold experts in one pass. `launch()` runs **plan → up → down → wait → gather**: plan decides
local expert/route slots and hands `B70_MOE_CPU_PCT` % of the cold experts to a host AVX-512 BF16 pool through a
host-USM mailbox; up computes `silu(x·Wg)·(x·Wu)`; down computes `w·(h·W2)`; wait polls the done flag with
uncached loads (graph-capturable); gather collects. 128 "cold worker" sub-groups overlap PCIe cold reads with
hot work, and each expert's weights are read once per layer. It exists because stock `XpuFusedMoe` costs ~56 µs
*fixed per call*, twice per layer, across 48 layers. It also ships `topk_softmax` replacing
`_moe_C.topk_softmax` (~12 µs fixed), which decides row validity on raw **bf16 bits** because `-ffast-math`
removes the float NaN tests.

**`b70-step-trace`** — measurement only. A mark is a single-work-item kernel that bumps a device counter, writes
a 32-bit code to a host-USM ring and publishes with a system-scope release; a poller pinned to core `23+8*dev`
stamps `CLOCK_MONOTONIC`. Needed because a graph replay is one opaque launch, `unitrace` hangs the 2-card engine,
the torch profiler kills it (`PTI_ERROR_NOT_IMPLEMENTED`), and the B70 has no `ext_oneapi_clock_device`. ~+1.5–2
ms per step. **Never built, not in the image.**

**`b70-hc`** — fused hyper-connection projections (`down` = GEMM + fused silu/hc; `up_mix` = up-GEMV + sigmoid +
4-stream mix with the gate never leaving registers), replacing two oneDNN GEMMs and two Triton kernels. No
README, no glue patch, no env var, no artifact — `docs/two-cards.md` records the gain as too small and it was
never wired in. **Never built.**

**Only `libb70_moe_decode.so` ships**, built by the image's `b70moe` stage (icpx 2026.0 from Intel's apt repo,
key sha256-pinned) into `…/layers/fused_moe/` with a build-time `torch.ops.load_library` assert — verified
present in the running image, 1 115 960 bytes, 2026-10-20 17:26. And `B70_MOE_DECODE_OP=0` live, so **it is
installed and disabled**. `libb70_step_trace.so` and `libb70_hc.so` do not exist anywhere on the box; the
kernels directory holds sources only.

`B70_MOE_GLUE` (patch **0051**) folds the 0047 route count into the plan kernel (`moe_decode_counted`, replacing
a `remainder` + `index_add_` pair) and swaps in `b70moe.topk_softmax` — ~1.5 % per decode step. It is documented
in `docs/switches.md` but exists only as `kernels/b70-moe-decode/vllm-0051-moe-glue.patch`, dev-tree only: not in
`patches/vllm/`, not in the image, not covered by `verify-tree.py`.

Tests (one B70, engine stopped, run inside the image): `test_moe_decode.py` compares against two stock
`XpuFusedMoe` calls at E 256/512, TOPK 10, N 640, K 2560, G 128, HOT 146, 48 layers inside an XPU graph;
`test_topk_align.py` checks id-exactness including ties, padding and NaN/±inf rows; `test_glue.py` checks
`moe_decode_counted` == `remainder` + `index_add_`; `test_hc.py` checks ulp diffs against oneDNN+Triton;
`test_step_trace.py` times a 1 GiB copy between marks against XPU events. `test_draft_head.py` needs a patched
dev tree via `DEV_TREE`.

## The Level Zero peer-residency shim

`image/shim/l0_peer_residency_shim.c`, 353 lines, wu1ff's MIT-licensed C, vendored because his GHCR image
`ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` disappeared. Workaround for intel/compute-runtime#968: without it
Level Zero mirrors peer device memory into host RAM, costing ~21 GiB on a 61 GiB host.

It is **not a symbol interposer** — it is a Level Zero **loader tracing layer**. The constructor
`setenv("ZE_ENABLE_TRACING_LAYER","1")`, `zeInit`, `zelTracerCreate/SetPrologues/SetEpilogues/SetEnabled`; it
hooks `zeContextMakeMemoryResident` (prologue) and `zeMemAllocDevice` / `zeMemFree` (epilogues), tracks
(context, base, size, owner), and keeps peer residency only for allocations ≤ `4 * 1024 * 1024` or ones whose
`backtrace()` shows `libccl.so.*` — everything else gets `*params->phDevice` **rewritten to the owner device**.
Counters plus a 15 s reporter; `_exit(127)` if the tracer fails to install. Built with
`gcc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror … -lze_loader -ldl -pthread`; source sha256 `f79e1bba…`
asserted at build time, output asserted equal to the upstream production binary `ae2c82f5…` — i.e. the vendored
source reproduces the lost binary bit for bit. 17 760 bytes, verified in the image.

## The kernels wheel

`vllm_xpu_kernels-0.1.14.1+b70.3-cp38-abi3-manylinux_2_28_x86_64.whl`, sha256 `f104a3e5…d589`, from the
`v0.30.0-b70.2` GitHub release. Fork `Lumnus/vllm-xpu-kernels` tag `v0.1.14.1+b70.3` = upstream `6d92b1bf` + 7
commits: our own int64 conv-state offset (fixes `DEVICE_LOST` past block id 5 042 — stock 0.1.14.1 without
`libgdn_index64.so` hits that), upstream #600 (ragged speculative traversal), #564 (UVA strides), #563
(unaligned vocab top-k/top-p), #578 (negative expert IDs in the XPU MoE remap), #586 (`atomic_buffer` init race),
and **B70-K1** (`swap_blocks_batch` H2D copies directly from pinned — which is why `B70_OFFLOAD_H2D_DIRECT=0`).
The top-level `patches/vllm-xpu-kernels/0001-…int64-state-offset.patch` is superseded by `b70.3/0001`, which is
the same change cherry-picked into the wheel.

## The `B70_*` switch surface

`docs/switches.md` is the canonical table; every b70 feature is env-gated and **off by default**, so with nothing
set the image is vLLM 0.30.0 plus the wu1ff layer. The full table is on haxor at
`~/AI/qwen38-flash-tp2/docs/switches.md`. What is live on this lane is listed in
[RECIPE.md](RECIPE.md#the-live-environment). Two traps in that surface:

- Older builds used a `LUMNUS_` prefix for the same switches; **those names are no longer read**, so a stale
  deployment runs with the feature silently off.
- A variable that is not in `docs/switches.md` is **ignored** — there is no error for a typo.

Ungated (active with every switch unset): 0009 accepts `"ultra"`; 0013 moves a KV-offload log line from DEBUG to
INFO and adds `is_pinned()`; 0014f records key positions and adds never-called policy hooks; **0018 chunks the
pinned CPU KV pool whenever one allocation is refused** (~31 GiB per rank on a B70); 0019 skips
`self_attn.indexer` tensors on dense-QSA configs; 0020, 0022, 0028, 0032.

## Known issues that shape the build

From `docs/known-issues.md`, the ones that bite:

1. An xe driver **job-timeout ban** after a GT reset — rebind the card
   (`echo BDF > /sys/bus/pci/drivers/xe/{unbind,bind}`); FLR does not help. Fix `347ccc0453fc` is 7.1-rc7,
   CVE-2026-90047 fixed in 6.18.51 / 7.2.5. After `DEVICE_LOST` bursts, **re-hash the weights and libraries** —
   the #966 host page-cache corruption survives a rebind. Re-apply fan tables.
2. Pinned CPU KV ≥ ~31 GiB per rank segfaults (torch 2.13) — patch 0018 chunks it.
3. A fresh Triton cache fails with `CL/cl.h: No such file` on intel-sycl-rt 2026.1.1 — needs Khronos OpenCL
   headers in `$VENV/include/CL/`.
4. Non-power-of-two CUDA-graph capture sizes trigger the token-1023 NaN "duct" loop.
5. The edge line (vLLM main + torch 2.14) has progressive decode corruption under concurrency — **do not use it**.
6. MTP NaN unless `--no-async-scheduling` **and** unpadded uniform draft sizes; grammar-trimmed drafts crash the
   stock 0.1.14 GDN kernel (fixed by kernels #600 / patch 0026). Note this lane runs **with** async scheduling and
   0026 on, and has run 345 M prompt tokens without a NaN event.
9. Full-vocab prompt logprobs spill ~3 GiB/rank and are never returned — 0029b.
10. ~0.8 GiB of host RAM leaks into the driver per engine stop/rebind (cause unknown; ~2 GiB with IOMMU on).
11. Stock 0.1.14.1 kernels without `libgdn_index64.so` → `DEVICE_LOST` past block id 5 042.
12. The release branch was never GPU-booted as written; AWQ was validated on an earlier branch.
13. First request after boot longer than one prefill chunk kills the MTP engine — see README.
14. **Do not use the torch profiler on this image** (`PTI_ERROR_NOT_IMPLEMENTED` kills the engine).

Also: `/health` and `/v1/models` stay 200 while the engine is wedged — probe with a 1-token completion and allow
≥180 s. And vLLM's own `--api-key` guards only `/v1,/v2,/inference,/cohere`; `POST /invocations` on this very
engine answered a real completion with no credential, which is why `b70_api_guard` exists.
