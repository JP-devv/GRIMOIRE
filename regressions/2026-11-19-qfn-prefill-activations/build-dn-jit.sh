#!/usr/bin/env bash
set -eo pipefail
# Invoked from the root of the isolated worktree by replay-dn.sh.
source "${QFN_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null
flags=(-fsycl -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -O2 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -Iinclude -Isrc)
bin=${1:?isolated binary path}
obj=${2:?fresh object directory}
mkdir -p "$obj"
sources=(tools/grimoire_main.cpp src/{grimoire,qwen35_loader,native_model,safetensors,quantize,gptq,gemv_decode,gguf,gguf_decode,gguf_qwen35,gemm_xmx,attention,deltanet,moe_kernels,tiered_moe,cpu_experts,moe_ref,ops,prefill,tokenizer}.cpp)
# Use existing JIT shared libraries unless a diagnostic supplies its own pair.
libs=${QFN_JIT_LIBDIR:-bin}
set -x
objects=()
for src in "${sources[@]}"; do
    o="$obj/${src##*/}.o"
    objects+=("$o")
    if [[ ! -f "$o" || "$src" -nt "$o" ]]; then
        icpx "${flags[@]}" -c "$src" -o "$o"
    fi
done
icpx "${flags[@]}" "${objects[@]}" -lpthread -L"$libs" -lgrimoire_gemm -lgrimoire_attn '-Wl,-rpath,$ORIGIN' -o "$bin"
"$bin" --help
sha256sum "$bin"
