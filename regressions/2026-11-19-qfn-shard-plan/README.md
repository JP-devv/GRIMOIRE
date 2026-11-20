# QFN shard plan geometry

The dense Flash-Next target has two logical KV heads on four ranks. Each rank needs a whole 256-wide head, so ranks 0/1 share head 0 and ranks 2/3 share head 1. Q and gate rows are interleaved per head, not stored in two tensor-wide halves.

The regression reconstructs the unsharded row and column order from the four plans, checks GDN head pairing and packed spans, and verifies unique expert ownership and `-1` for non-local or invalid IDs. It covers both shared-expert splits (BF16 `160/160/160/160`, g128 `256/128/128/128`) and rejects unsupported geometry, including QSA and invalid ranks. It uses synthetic coordinates, not checkpoint weights, and does not qualify engine execution or tensor storage layouts.

Run from the GRIMOIRE root with a host compiler (no SYCL or GPU):

```sh
(
    set -eu
    exe=$(mktemp /tmp/test_qfn_shard_plan.XXXXXXXX)
    trap 'rm -f "$exe"' EXIT
    c++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
        regressions/2026-11-19-qfn-shard-plan/test_qfn_shard_plan.cpp -o "$exe"
    "$exe"
)
```
