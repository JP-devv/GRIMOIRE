#!/usr/bin/env bash
set -euo pipefail
B=$(cd -- "$(dirname -- "$0")/../.." && pwd)
D=$(mktemp -d)
trap 'rm -rf -- "$D"' EXIT
write_rank() {
    local rank=$1 pp=$2 decode=$3 frozen=${4:-good} accepted=${5:-21}
    {
        printf 'prompt ids (2706): %s\n' "$(< "$B/prompt.ids")"
        if [[ $frozen == good ]]; then awk '1' "$B/frozen.ids"; else printf 'greedy ids (32): wrong\n'; fi
        printf 'generation phases: prefill_tokens=2706 prefill_ms=%s decode_tokens=31 decode_ms=%s\n' "$pp" "$decode"
    } > "$D/r$rank.log"
    printf '  QFN TP4 synchronization: socket fan-out\n  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum\n' > "$D/r$rank.out"
    printf 'generated text  spec: MTP depth 3 -- 1.91 accepted/step (%d of 33 drafted, 63.6%%) over 11 steps\n' "$accepted" >> "$D/r$rank.out"
}
score() { bash "$B/combined-c1-score.sh" "$D"; }
for rank in 0 1 2 3; do write_rank "$rank" 676.000 221.000; done
write_rank 0 676.500 221.000
write_rank 3 676.000 221.428
score | jq -e '.targets_pass and .cold_prefill_tok_s == 4000 and .decode_tok_s >= 140 and .mtp3.accepted == 21' >/dev/null
write_rank 3 676.000 221.429
score | jq -e '(.targets_pass|not) and .decode_tok_s < 140 and .cold_prefill_tok_s == 4000' >/dev/null
write_rank 3 676.501 221.000
score | jq -e '(.targets_pass|not) and .decode_tok_s >= 140 and .cold_prefill_tok_s < 4000' >/dev/null
write_rank 3 676.000 221.000 bad
if score > "$D/bad.json" 2> "$D/error"; then echo 'accepted different frozen IDs' >&2; exit 1; fi
grep -Fq 'Frozen IDs disagree' "$D/error"
[[ ! -s $D/bad.json ]]
write_rank 3 676.000 221.000 good 20
if score > "$D/bad.json" 2> "$D/error"; then echo 'accepted different MTP3 counters' >&2; exit 1; fi
grep -Fq 'MTP3 acceptance mismatch on rank 3' "$D/error"
[[ ! -s $D/bad.json ]]
write_rank 3 676.000 221.000
printf 'QFN packed N64: trace forbidden in timed rank\n' >> "$D/r3.log"
if score > "$D/bad.json" 2> "$D/error"; then echo 'accepted traced timings' >&2; exit 1; fi
[[ ! -s $D/bad.json ]]
printf 'Combined C1 CPU scorer PASS (both targets, either missed target, frozen ID / MTP3 / trace rejection)\n'
