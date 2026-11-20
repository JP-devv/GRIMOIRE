function end_round() {
    if (layers != 12) bad = 1
    print context
    ++rounds
    layers = 0
    delete seen
}
/^QFN verifier KV:/ {
    if (NF != 7 || $4 != "rank=" rank || $5 !~ /^layer=[0-9]+$/ ||
        $6 !~ /^context=[0-9]+$/ || $7 != "splits=" splits) { bad=1; next }
    layer = substr($5, 7) + 0
    pos = substr($6, 9) + 0
    if (layers && pos != context) {
        if (pos <= context) bad = 1
        end_round()
    }
    if (pos < 2048 || layer >= 48 || seen[layer]++) bad = 1
    context = pos
    ++layers
}
/^Verify attention:|^generation phases:|fenced\)/ { bad = 1 }
END {
    if (layers) end_round()
    if (bad || rounds != steps || !rounds) exit 1
}
