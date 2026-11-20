#!/usr/bin/env bash
# CPU-only check, or bounded same-build prefill comparison in a new GPU window.
set -euo pipefail
export LC_ALL=C
cd -- "$(dirname -- "$0")"
B=$PWD
P=/tmp/qfn-catchup.pf1/packed-n64/qualification
ARCHIVE=$P/build-collection/retry-001
REFERENCE=/tmp/qfn-catchup.pf1/verifier/gate-ZwV33FAa
BASE=/home/haxor/AI/GRIMOIRE-qfn-qgate/diagnostics
SERVING=$BASE/act-xe2-tilecut-ab-6IU2nP/serving.sha256
BIN=$P/artifacts/qfn-packed-n64
mode=${1:-}
[[ ($mode == check && $# == 1) || ($mode == run && $# == 3) ]] || {
    printf 'Usage: %s check | run EMPTY_DIR APPROVED_DEADLINE_EPOCH\n' "$0" >&2
    exit 2
}
check_reference() {
    sha256sum -c packed-prefill.sha256
    sha256sum -c inputs.sha256
    sha256sum -c native.sha256
    sha256sum -c packed-upstream.sha256
    sha256sum -c packed-baseline.sha256
    sha256sum -c "$SERVING"
    python3 -B checkpoint-lock.py
    cmp prompt.ids "$BASE/act-moe-fU6ZtS/prompt.ids"
    for rank in 0 1 2 3; do
        cmp frozen.ids "$BASE/act-xmx-ab-N8CMZj/grouped-gate/matched-r$rank.ids"
    done
}
check_prepared() {
    local result
    if [[ ! -r $ARCHIVE/full-build.receipt ]]; then
        printf 'Not ready: retry-001 full-build.receipt is absent; no action taken on the collector.\n' >&2
        return 3
    fi
    result=$(< "$ARCHIVE/full-build.receipt")
    if [[ $result != 'stage=verify exit=0' ]]; then
        printf '%s\n' "$result" >&2
        return 3
    fi
    if [[ ! -r $ARCHIVE/SHA256SUMS || ! -r $P/artifacts/prepare.exit ||
          ! -r $P/A-B-BINARY.sha256 || ! -x $BIN ]]; then
        printf 'Not ready: successful build receipt present, but prepared candidate is incomplete.\n' >&2
        return 3
    fi
    [[ $(< "$P/artifacts/prepare.exit") == 0 && $(realpath -e -- "$BIN") == "$BIN" ]]
    [[ $(< "$P/A-B-BINARY.sha256") == "$(sha256sum -- "$BIN")" ]]
    sha256sum -c "$ARCHIVE/SHA256SUMS"
    sha256sum -c "$P/artifacts/inputs.sha256"
    sha256sum -c "$P/artifacts/base.sha256"
}
check_reference
if [[ $mode == check ]]; then
    python3 -B packed-prefill.py reference "$REFERENCE"
    check_prepared
    exit 0
fi
D=$(realpath -e -- "$2")
deadline=$3
[[ -d $D && -z $(ls -A -- "$D") && $deadline =~ ^[1-9][0-9]{9}$ && -n ${TMUX:-} ]]
stage=collector-artifacts
finish() {
    local status=$?
    trap - EXIT
    if ! sha256sum -c "$SERVING" > "$D/serving-after.sha256"; then stage=serving-check; status=1; fi
    printf 'stage=%s exit=%d\n' "$stage" "$status" > "$D/receipt"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT
check_prepared
stage=deadline
(( deadline - $(date +%s) >= 720 )) || { printf 'Need at least 12 authorized minutes\n' >&2; exit 124; }
cp "$P/A-B-BINARY.sha256" "$D/"
cp native.sha256 packed-baseline.sha256 packed-upstream.sha256 "$D/"
mkdir "$D/arms"
stage=packed-ab
bash "$P/packed-ab.sh" run "$D/arms" "$deadline" > "$D/packed-ab.log" 2>&1
stage=verify-inputs
check_reference
check_prepared
cmp "$D/A-B-BINARY.sha256" "$P/A-B-BINARY.sha256"
stage=score
python3 -B packed-prefill.py compare "$D/arms" "$REFERENCE" > "$D/comparison.pending.json"
mv "$D/comparison.pending.json" "$D/comparison.json"
stage=verify
