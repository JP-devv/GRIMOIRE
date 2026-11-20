# QFN paired results and next change

Both final programs ran inside the exclusive 2026-11-20 12:24:24–12:39:24 host-local window, acquired `/tmp/grimoire-qfn-gpus.lock` and passed the four-device idle check before launch. No serving process was stopped or replaced. The serving checksum remained `61ce37eb67dc65162e265302a5b424f7659a358e42f131a611cd4675d8cc8cd1`.

## Unprofiled throughput

`build/receipt` and `verifier/gate-ZwV33FAa/receipt` each say `stage=verify exit=0`. On every timed arm, all four ranks matched all 32 frozen output IDs. Each dense arm additionally proved the named attention backend for every full-attention layer in both chunks.

| Experiment | Control prompt tok/s | Candidate prompt tok/s | Control output tok/s | Candidate output tok/s |
|---|---:|---:|---:|---:|
| Dense XMX pair 1 | 1,063.34 | 991.84 | 24.77 | 47.30 |
| Dense XMX pair 2 (reversed) | 1,069.56 | 1,164.34 | 45.50 | 47.46 |
| MTP3, off/on | 1,052.33 | 904.58 | 48.92 | 29.23 |

XMX prefill latency changed +7.21% and −8.14%; there is no repeatable win. The first dense control also had an output-rate outlier, so do not attribute that variation to a prefill-only switch. MTP3 accepted 21/33 drafts, but reduced output rate by 40.2%. Keep both experiments off.

The first dense launch failed with `100ss` and no GPU work; the corrected numeric timeout passed. The first MTP3 launch failed at `mtp-on-gate` because the prompt-only `m32` overlay is refused on verifier rows. The retry omitted `m32` and DecodeM8 from both MTP arms, preserving the same settings for the control and candidate. Initial failure logs live in `dense-timeout-launch-error/` and `verifier/gate-exMsrFij/`.

## Fenced, separate profiles

The MTP3 profile has 11 four-row rounds per rank. Median full attention is 44.8 ms/round, MoE 24.5 ms, HC 5.9 ms, DeltaNet 5.5–5.6 ms, local vocabulary head 2.56–2.58 ms, rank agreement 0.02–0.05 ms and draft cache warm 0.50 ms. The fenced per-step mean for the full verifier is 96.40–96.63 ms; the full phase accumulation and fenced mean are distinct statistics. Focus future decode work on verifier attention before batching the vocabulary head or rewriting draft warm.

On the same frozen prompt, prefill MoE is the largest cumulative measured phase (415.8 + 557.5 ms in the fenced MTP3 prompt; includes reductions). The independent parity-passing `act-xe2-profile-0zms3D/profile-m8` trace localizes 378.061 ms to routed-device work on the 658-row tail, versus 144.020 ms for the larger 2,048-row chunk using the qualified Xe2 kernel. The tail is the first prefill target; the profile includes staging, expert GEMMs, and data movement and does not identify their individual costs.

## Implemented, not GPU-qualified

`packed-n64.patch` applies to `GRIMOIRE-qfn-qgate`; `packed-n64/` is the isolated copy used for the build. `GRIMOIRE_QFN_GROUPED_PACKED_N64=1` stages both INT4 nibbles with one byte/scale/zero read in the existing routed N64 SYCL kernel. The `test_qfn_packed_n64` host fixture passed. Full B70 build outputs are `packed-n64/full-build.{log,receipt,exit}`. This new path was not run on the devices. Qualify actual expert output and model parity before benchmarking or enabling it anywhere else.
