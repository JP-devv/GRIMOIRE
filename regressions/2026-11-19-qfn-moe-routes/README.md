# QFN resident INT4 route/gather contract

From the repo root, build and run the standalone CPU regression (no SYCL or model files):

```sh
(
    set -eu
    exe=$(mktemp)
    trap 'rm -f "$exe"' EXIT
    c++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
        regressions/2026-11-19-qfn-moe-routes/test_qfn_moe_routes.cpp -o "$exe"
    "$exe"
)
```

The test compares stable expert-major ordering to a per-expert scan and the
weighted gather to a direct sum over original slots. Its top-10 fixture includes
a token whose only contributions are in slots 8 and 9, an all-padding token, and
both real expert 0 and `-1` padding.

## Existing interfaces

- `launch_permute_rows_bf16` (`src/kernels.hpp`, `src/prefill.cpp`) takes an
  `int32_t` token index per permuted row. Use `perm_token` with the **active** row
  count, not `rows * top_k`.
- `grimoire_xe2_grouped_w4a16` (`src/xe2_grouped_bridge.hpp`) takes a count per
  expert as `const int*` (`int32_t` on this build). The generic host branch in
  `src/grimoire.cpp` passes dense expert-order counts with `expert_ids == nullptr`.
  The plan keeps this order, including zero counts. Its prefix offsets also have
  a final entry equal to the active row count.
- The FP32 `launch_moe_unpermute` already uses an `int32_t` inverse index per
  original slot, the actual `top_k` stride and weights, and skips negative inverse
  indices. The existing `launch_moe_unpermute_bf16` instead reads exactly eight
  slots without checking for `-1`. It cannot consume top-10 or padded plans as-is.
  A compatible BF16 gather must iterate the actual `top_k`, skip absent slots
  before reading compacted output, and write zero for a token with no routes.

No engine integration or GPU behavior is tested here.
