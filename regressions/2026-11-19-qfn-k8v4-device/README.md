# K8/V4 dense-decode — DEVICE PROTOTYPE (prototype, not a runtime component)

**Prototype.** One self-contained SYCL translation unit that runs the K8/V4
paged dense decode over **already-packed** page bytes and compares it against a
host scalar fp64 reference that reads the **same bytes**. Nothing from the
runtime is used, linked, or modified; nothing is tuned. It is a fixture, not a
kernel to ship.

Target geometry: `D=256`, `GQA=6` (one KV head, six Q heads — the TP4 per-rank
slice), `PAGE=64`, int8 K, packed int4 V (even dimension in the low nibble),
fp32 `k_scale`/`v_scale`/`v_zero`, region-major page order `K,V,k_scale,
v_scale,v_zero`, 396 B per token head, 25,344 B per page.

The packed bytes, the quantizers, the causal lens and the masked-tail validity
rule are the documented K8/V4 rules as captured by the host fixture
`regressions/2026-11-19-qfn-k8v4/test_qfn_k8v4.cpp` (read only, unchanged, not
linked). This file reimplements them and re-checks them against that fixture's
published numbers before any device comparison is trusted.

## Files

| file | role |
| --- | --- |
| `test_qfn_k8v4_device.cpp` | packer, fp64 packed-byte reference, 4 SYCL kernels, all checks |
| `build.sh` | build + link for JIT (`-fsycl-targets=spir64`); builds only, runs nothing |
| `build/test_qfn_k8v4_device` | the artifact |

## Build

```sh
bash regressions/2026-11-19-qfn-k8v4-device/build.sh
```

`icpx 2026.2.0`, `-fsycl -fsycl-targets=spir64 -O2 -std=c++20 -Wall -Wextra
-Werror`, rpath baked to the SYCL runtime beside the `icpx` that built it. Builds
in ~20 s, compiles device images, needs no GPU and touches no render node.

## Run

```sh
export OCL_ICD_FILENAMES=""; set +u; source /opt/intel/oneapi/setvars.sh --force
B=regressions/2026-11-19-qfn-k8v4-device/build/test_qfn_k8v4_device

$B --host-only                 # packer + fp64 reference, NO queue, NO device memory
$B --list-devices              # enumerate (needs the oneAPI env for adapter init)
$B --device B70                # device run: one free card, < 1 MiB device memory
$B --cpu                       # optional: same kernels on the SYCL CPU device
```

The binary needs the oneAPI environment sourced for UR adapter initialisation
(the baked rpath only covers `libsycl.so.9`). Without it: `UR adapter
initialization failed: 37` and no devices.

## Device path

Four command groups, all reading the one packed buffer:

1. **score** — `nd_range` one 256-lane work-group per (pair, page, token); int8
   dot via `reduce_over_group(plus)`. `k_scale` is read and the score formed for
   **all 64** page tokens, and only then overwritten with `-INFINITY` when
   `tok >= n_valid` or `k_abs >= end`.
2. **softmax** — per page, 64-lane group max + sum, probabilities rounded
   through `sycl::half`.
3. **pv** — per page, one lane per output dimension, fp16 dequantized V, fp32
   accumulator. Rows outside `n_valid` are skipped **before** any affine
   metadata or nibble is read.
4. **merge** — running M/L rescale across pages in fp32, one work-item per pair.

## What is checked

- Layout: page 25,344 B, slot 396 B, region offsets 24576/24832/25088/25344,
  K token stride 256 B, V token stride 128 B, page stride = whole page, a 1e-7
  fp32 scale round-tripping byte for byte.
- Packer identity against the host fixture's published numbers: K tie row
  `[127 1 -1 3 2]` (half-away, scale exactly 1.0), V ladder `even | odd << 4`
  (`byte[0] == 0x10`), flat `+0.5` (scale floor 1e-8, zero 0, `0xFF`, dequant
  1.5e-7), flat `-0.5` (zero 15, `0x00`), all-negative row (zero clamps to 15,
  one code survives), zero K row (scale 0), causal lens `[135 136 137]`,
  10-token tile refusal.
- Seven cases, including **contexts crossing page boundaries**: `seq=137,
  q_len=3` (pages 64/64/9, lens 135/136/137), `seq=65, q_len=2` (crosses by one
  token), `seq=128` and `seq=192, q_len=10` (full pages, max decode tile),
  `seq=1`, `seq=0`. Each: packed-byte hash printed, output finite, reference
  finite, `max |device - reference| < 5e-3` (derived fp16-prob/fp16-V bound),
  `max |device score - fp64 score| < 1e-4` on visible tokens, and the packed
  bytes proved unmutated by the run.
- Finite-error handling: `seq0` yields exactly zero rows; `seq1` returns the
  decoded V row; no NaN may reach an output.
- **Negative controls** on the crossing case: the NaN/inf-poisoned masked tail
  (K `0x7F`, V `0xFF`, NaN `k_scale`, inf `v_scale`, `1e30` `v_zero`) must
  change the device output, the device scores and the fp64 reference by exactly
  zero; poisoning a **live** row must move the output (> 1e-3). The mask
  boundary is tested from both sides (token 136 moves, token 137 does not), page
  2 row 0 (token 128) must move, and on `seq=65` token 64 moves while token 65
  does not.

## Status of this prototype (honest)

- `--host-only`: **passes, exit 0, 0 failures** on haxor (icpx 2026.2.0, i7-13700K).
- The device path was **not executed here**: the GPUs are in use by other tests,
  so no card was touched. Build and link are verified; the binary is ready.
- The optional `--cpu` path currently **crashes (SIGSEGV, exit 139)** inside the
  Intel CPU device runtime immediately after the USM upload of the first case,
  i.e. during/at the first score kernel. Bisection so far: plain USM copies,
  `single_task`, `nd_range` + `reduce_over_group(plus)` on int32,
  `reduce_over_group(maximum)` + `sycl::exp`, `sycl::half` store/load/convert,
  and a 256-lane int8 dot kernel with a mask all run fine on that device in an
  isolated probe (`/tmp/probe.cpp`, stages 1-6 all exit 0). The cause is **not
  yet isolated** and is not attributed to any specific construct. This says
  nothing about the GPU path, which was never run.
- Do not treat this fixture as validation of the shipped kernel or of the
  runtime; it only pins the documented layout/formula semantics against one
  independently wired scalar reference. No measurements from a card exist yet.
