#!/usr/bin/env bash
# CPU only: validate and score one four-rank combined C1 receipt.
set -euo pipefail
export LC_ALL=C
B=$(cd -- "$(dirname -- "$0")" && pwd)
checks=/tmp/qfn-catchup.pf1/small-moe/qualification
D=${1:?rank logs and outputs directory}
reference=
for rank in 0 1 2 3; do
    stem=$D/r$rank
    grep -Fxq '  QFN TP4 synchronization: socket fan-out' "$stem.out"
    grep -Fxq '  TP4 P2P: direct peer-push, FP32 payloads and rank-order sum' "$stem.out"
    awk '/^QFN packed N64:|^QFN M8 decode:/{bad=1} END{exit bad}' "$stem.log"
    awk -f "$checks/check-timing.awk" "$stem.log" "$stem.out" >/dev/null
    acceptance=$(awk -f "$checks/check-spec.awk" "$stem.out")
    if (( rank == 0 )); then reference=$acceptance; fi
    [[ $acceptance == "$reference" ]] || { echo "MTP3 acceptance mismatch on rank $rank" >&2; exit 1; }
done
IFS=$'\t' read -r steps accepted drafted <<< "$reference"
phases=$(python3 -B "$B/c1.py" native "$D"/r{0,1,2,3}.log)
jq --argjson steps "$steps" --argjson accepted "$accepted" --argjson drafted "$drafted" \
    --argjson decode 140 --argjson prefill 4000 '
    . + {mtp3:{steps:$steps,accepted:$accepted,drafted:$drafted},
         targets:{decode_tok_s:$decode,cold_prefill_tok_s:$prefill},
         targets_pass:(.decode_tok_s >= $decode and .cold_prefill_tok_s >= $prefill)}' <<< "$phases"
