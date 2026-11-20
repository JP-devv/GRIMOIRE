# QFN C1 integration on haxor

The integration worktree is `~/AI/GRIMOIRE-qfn-integration` on `qfn-serving-integration`. Its base is qgate commit `ec89276f0fe0b257f17899170d65989a54ba4880` plus the qgate working-tree overlay. The two complete staged source sets and their manifest hashes are under `build-inputs`. The installed reference image ID is `sha256:bcc91a8966e805a632f575b0467d4b8f926e1f6d7ec36cf4ecc16f3627537a05`.

## Rebuild and verification

On haxor, build into a new directory using the clean sycl-tla checkout at commit `87f6850680a580654b9ea2c80dbc01aeb36ad231`.

```bash
make test-correctness
bash qfn-enhanced/build-inputs/rebuild.sh combined-c1 /path/to/new-build \
    "$HOME/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-O5gnlh/sycl-tla"
```

The clean build reproduced the staged CLI and both G31 AOT libraries byte for byte. The engine's general objects target SPIR64. The additional native HTTP link consumes those same objects, has a local-library RUNPATH, and adds raw-ID completions for frozen requests. `build-inputs/combined-c1/rebuilt.sha256` pins both entry points and libraries. This build retains the serving admission guard for TP4 MTP. The CLI diagnostic's MTP3 scores below are separate from HTTP.

## Frozen receipt status

| Build and configuration | Generated tok/s | Cold prompt tok/s | Decode ms remaining to 140 | Prefill ms remaining to 4,000 |
|---|---:|---:|---:|---:|
| Native baseline, MTP off | 46.691528650524454 | 1022.9552519452594 | 442.503428571 | 1968.777 |
| Combined native CLI diagnostic, MTP3, small-M MoE and packed-N64 | 21.46615203298309 | 957.064380086836 | 1222.705428571 | 2150.896 |

Both native arms reproduced the 32 pinned IDs across all four ranks. Combined MTP3 accepted 21 of 33 drafts over 11 rounds. The runs use different configurations. The combined run does not isolate gains or regressions from either switch. The native baseline overlapped the tail of a CPU rebuild during startup and has no reverse-order passing replicate.

The first installed reference attempt reached oneCCL initialization and exposed a missing `/dev/dri/by-path` mount. The reverse-order run added the mount. It reached API readiness, then failed its first real request when FP8 PLE tried to synchronize a queue during XPU graph recording. The eager MTP-off attribution run used the same image and checkpoint roles with graph capture disabled. It confirmed 2,706 matching input IDs, 32 generated IDs and zero cached tokens. Frozen output diverged at token 23 (native 20438, reference 9468), so the correctness gate rejected the arm. Raw diagnostic phase counters were 0.7285571990069002 s prefill and 2.2011744880583137 s decode, with engine-request TTFT at 0.7401349544525146 s. These counters do not establish parity. See the recorded invocation and kernel journals under `receipts`.

## Scope and restore

The runners use a loopback endpoint, the existing four-GPU lock and a deadline with teardown margin. They verify idle device nodes and the existing serving binary checksum. Test workers and test containers are scoped to the window. The monitor wrapper restores the original tmux pane's `nvtop`. Rollback is stopping the isolated run and verifying idle device nodes before monitor restore. No production symlink, executable or lane configuration is changed.

## Repeated native A/Bs

| Experiment | Same-build control | Candidate | Frozen gate |
|---|---|---|---|
| Split-KV MTP3 decode, forward pair | 29.128028962656927 tok/s | 51.124828690594185 tok/s | All ranks, 32 IDs, 21/33 acceptance |
| Split-KV MTP3 decode, reverse pair | 28.851299471835084 tok/s | 50.53098282761599 tok/s | All ranks, 32 IDs, 21/33 acceptance |
| FP32 RSAG prefill, forward pair | 2,516.089 ms | 2,211.002 ms | All ranks, 32 IDs, 97 full-chunk RSAG calls per rank |
| FP32 RSAG prefill, reverse pair | 2,356.040 ms | 2,174.658 ms | All ranks, 32 IDs, 97 full-chunk RSAG calls per rank |
| HC activation fusion, forward pair | 2,429.226 ms | 2,437.381 ms | All ranks, 32 IDs, 94 fused conversions per rank |
| HC activation fusion, reverse pair | 2,355.674 ms | 2,408.302 ms | All ranks, 32 IDs, 94 fused conversions per rank |

The best RSAG prefill rate is 1,244.3335917647742 tok/s and it needs another 1,498.158 ms on the frozen prompt. Its same-build MTP-off decode was 48.93796717998911 tok/s on that arm. Split-KV and RSAG measurements above come from separate pinned diagnostic builds. No speed sum is claimed. Native HTTP raw-ID smoke on the base build passed all 32 IDs, a 3.167140727979131 s non-streamed completion and first visible text in 2.152708865934983 s in the separately streamed request.

The original sequential replay receipt reports recurrent, convolution and logit differences after partial rejection despite matching continuation IDs. New fixed-proposal replay matched the first-row and four-row predicted IDs. Serving TP4 MTP admission remains gated by the full sequence-state checks, including each rejection position, full acceptance, EOS, and cancellation. The 32K benchmark context does not establish 262K serving-capacity parity.

## Additional fixtures and pinned reference

The 2,706-to-256 MTP-off pair reproduced every output ID on all four native ranks and on the installed eager reference, including the 32 frozen prefix IDs. Reference cached prompt tokens were zero. Native measured 48.665310908358066 generated and 1,087.3308278380803 cold prompt tok/s. Reference measured 14.150761524414197 generated and 3,762.3715286367856 cold prompt tok/s. Reference engine-request TTFT was 0.7330436706542969 s. This is an attribution pair with the long benchmark CLI's explicit EOS-independent output budget.

The 2,048-ID full-chunk isolation used those same stop rules on the native repeat. It reproduced all 2,048 input IDs and emitted 32 on all four ranks, with native prefill at 1.4354 s. The corresponding installed eager reference returned the same input IDs and zero cache hits with 0.40110780904069543 s engine prefill, but the outputs diverged at generated token 10 (native 271, reference 198). Its raw phases are diagnostic and fail the pair's exact output gate. The earlier default-EOS native arm stopped immediately and is a separate rejected receipt.

The installed 262K full-and-piecewise TP4/EP4 MTP3 recipe reached readiness and returned HTTP 500 on the first real request. The worker failed in FP8 PLE pre-forward when synchronizing an XPU queue that was recording a graph. The separately tested eager MTP3 arm passed all 32 frozen output IDs and zero prefix hits, measuring 18.21143244917593 generated and 3,471.5888726897842 cold prompt tok/s. The workers confirmed 1,024 + 1,024 + 658 input rows. It accepted 22 of 27 proposals in nine completed speculative rounds, with 8/7/7 acceptances by proposal position. The native speculative diagnostic accepted 21 of 33 and the different proposed work does not establish cross-engine acceptance parity.

## Prefill work removed

Native causal tile trimming with the FP32 RSAG path enabled saved 203.086 and 41.348 ms in the forward and reverse frozen pairs. The faster arm was 2,141.950 ms prefill, 1,263.3348117369687 prompt tok/s and 48.672344060089934 generated tok/s. The controls had wide spread. Each arm reproduced all frozen IDs. The two chunks each dispatched the trimmed dense path on all 12 layers.

The separate persistent packed HC up/mean fusion with RSAG enabled saved 52.216 and 86.866 ms in reversed frozen pairs. It passed the frozen IDs and verified 96 fused HC mixers per rank in the 2,048-row chunk. Its faster arm was 2,104.990 ms prefill, 1,285.5167958042557 prompt tok/s and 49.01131528979126 generated tok/s. This arm keeps the original BF16 weights and FP32 gates and requires 600 MiB of resident packed weights per rank. The combined build is measured separately, as no timing savings are added across these binaries.

## Combined native prefill candidate

The one-build frozen C1 controls and both feature-on arms used FP32 RSAG. The feature-on arms additionally enabled causal key-tile trimming and persistent packed HC up/mean fusion. Both orders passed all 32 frozen IDs and each stage's dispatch count. Causal trimming ran on all 12 dense layers in both chunks. The HC fusion ran 96 times per rank in the full chunk.

| Order | Control prefill (ms) | Candidate prefill (ms) | Saved (ms) | Candidate generated tok/s | Candidate cold prompt tok/s |
|---|---:|---:|---:|---:|---:|
| Control then candidate | 2265.573 | 2054.576 | 210.997 | 49.63732727010713 | 1317.060064947707 |
| Candidate then control | 2202.568 | 2053.170 | 149.398 | 49.08845898659252 | 1317.9619807419745 |

The fastest prefill remains 1,376.670 ms over the 676.5 ms target. On that same arm, decode is 631.513 ms for 31 timed tokens, leaving 410.084428571 ms to remove for 140 tok/s. The benchmark uses MTP off on the native serving candidate. All 512 routed experts remain assigned across the four GPU ranks.

An uncontended replay of the separate integrated RSAG + split-KV diagnostic passed every frozen ID and 21/33 acceptance at 49.566453424322425 generated and 1,026.6360066970358 cold prompt tok/s. This supersedes the CPU-build-overlapped diagnostic speed sample for that configuration. It is still a diagnostic CLI. The isolated small-M MoE variant dispatched on all 48 routed layers for each four-row verifier call but changed the frozen output at token 23 and changed acceptance to 22/29 over ten rounds. Its several-dozen-millisecond total reduction followed a lower round count. Both the control and rejected trial spent roughly 60.5 ms per completed round. Keep small-M disabled.

Rebuild the measured serving candidate from the pinned base and overlays:

```bash
bash qfn-enhanced/build-inputs/rebuild.sh prefill-combined /path/to/new-build \
    "$HOME/AI/GRIMOIRE-qfn-qgate/diagnostics/act-xe2-O5gnlh/sycl-tla"
```

The unprofiled engine-phase data is under `receipts/frozen-pair.{xnvQ6QCQ,RBKzQByx,V391O9oi,XuHOWmlJ}`. The raw-ID HTTP test measures first visible streamed text separately and uses a local-only endpoint. The combined candidate has a native C++/SYCL execution path. The process mappings show the local GRIMOIRE libraries, oneAPI SYCL/UR, and Level Zero. Benchmark clients and the vLLM reference instrumentation are separate tooling.

### Pinned native serving package

The verified executables and shared libraries are in
`artifacts/prefill-combined/` on haxor. Verify them with
`(cd artifacts/prefill-combined && sha256sum -c SHA256SUMS)`.
Use the `prefill-combined` rebuild invocation above to reproduce the package.
`serving.env` records the effective GRIMOIRE settings. The four-rank
device placement, socket setup, model argument, Level Zero selector and
exclusive-window cleanup are in `bench/c1-frozen/run-native-http.sh`
(relative to this document). Use variant `prefill-combined` and context
`262144` to reproduce the HTTP smoke.

The final build passed the frozen HTTP gate with one sequence slot at a
262,144-token configured context. The streamed first visible text took
1.820441753952764 s. A separate nonstreaming request completed in
2.75019331893418 s. This smoke used the 2,706-token fixture.
See `../receipts/native-http.TbGsC5iI/` for the exact build pins,
effective flags, rank logs, and restoration check.
