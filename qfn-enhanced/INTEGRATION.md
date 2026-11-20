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
