# QFN INT4 MoE kernel regression

This synthetic fixture exercises the native prompt MoE at `M=32`, `E=512`, `H=2560`, `I=640`, top-10 and INT4 groups of 128. It runs the resident GEMV baseline, per-expert XMX and grouped XMX. It does not load a checkpoint or run the model's router or shared expert.

## Build and run

On a host with the engine's JIT objects and bridge libraries built:

```sh
source /opt/intel/oneapi/setvars.sh
QFN_GEMM_XMX_OBJ=/path/to/current/gemm_xmx.o \
    bash regressions/2026-11-19-qfn-moe-gpu/build.sh
regressions/2026-11-19-qfn-moe-gpu/build/test_qfn_moe_gpu --host-only
# Exclusive GPU window; runner checks the render nodes before dispatch.
bash benchmarks/qfn/run-qfn-gpu-fixture.sh /tmp/qfn-gpu-fixture --force-dpas
```

Omit `QFN_GEMM_XMX_OBJ` if `obj-jit/gemm_xmx.o` already includes the grouped kernel. The host-only run checks the route plan and independently checks packed INT4 row values without creating a queue. The device run uses about 1.3 GiB of VRAM.

## What it checks

- Sparse deterministic weights have one nonzero in each 128-wide group. They cover both affine and signed (`0xff` sentinel) INT4; the affine background nibble equals the zero point, not zero. Dense activations and non-power-of-two BF16 scales exercise rounding.
- Routes include an empty token, holes in the first eight slots, high expert IDs, a 76-row expert spanning multiple 32-row tiles, and experts with ragged tails. The empty output row must be exactly zero.
- The independent double reference reads no packed bytes. The FP32-facing version checks GEMV; the version that explicitly rounds matrix operands and the SwiGLU output to BF16 checks grouped XMX. Nonfinite output fails the gate. `--force-dpas` also makes the grouped and per-expert results compare exactly; the default mixed dispatcher can use an INT8-activation GEMM for the 76-row expert, so differences in that comparison are diagnostic only.

## TP4 rank-local GEMV gate

The device run also emulates rank 1 on one B70 with exactly 128 resident experts. It checks that production remaps `[127, 128, 255, 256]` to `[-1, 0, 127, -1]`, then dispatches the same batched GEMV kernels as prefill. The unsharded reference masks only nonlocal IDs and uses the original router weights, without renormalizing. It checks a single row and a three-row batch (including a row with no local routes and a row with experts in slots 8 and 9). Swapping the two differently packed endpoint experts must change the output.

Both rank-local outputs matched the masked full-table reference exactly on one B70, and swapping the endpoints changed an output by up to `0.00167755`. This checks the routed contribution, not the four-card reduction or attention path.

On B70 G31 with SYCL 2026.2, the grouped comparison against the BF16 reference had max absolute error at most `1.192e-7` across the three weight-layout cases. With `--force-dpas`, grouped and per-expert outputs agreed exactly. These are kernel checks, not a guarantee of model-level parity against the FP32-facing baseline.
