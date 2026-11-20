# Strictly check the active QFN path and pair events with the fenced phase in
# the same logged verifier round. Do not substitute a different kernel.
function fail(message) {
    printf "%s:%d: %s\n", FILENAME, FNR, message > "/dev/stderr"
    bad = 1
}
/^Verify attention: / {
    if (is_split) {
        if (NF != 14 || $3 != "backend=batched" || $4 != "rows=4" || $5 !~ /^context=[0-9]+$/ ||
            $6 != "splits=32" || $7 != "kv_type=e4m3" || $8 != "k_layout=dim-major" ||
            $9 != "v_layout=token-major" || $10 != "q_heads=6" || $11 != "kv_heads=1" ||
            $12 != "head_dim=256" || $13 !~ /^split_kv_ns=[1-9][0-9]*$/ || $14 !~ /^reduce_ns=[1-9][0-9]*$/) {
            fail("bad split verifier attention event"); next
        }
        pos = substr($5, 9) + 0
        if (!layers) context = pos
        if (await_head || pos != context) fail("split round mismatch")
        ++layers
        ns += substr($13, 13) + 0
        reduction += substr($14, 11) + 0
        next
    }
    if (NF != 16 || $3 != "backend=block" || $4 != "rank=" rank ||
        $5 !~ /^layer=[0-9]+$/ || $6 != "rows=4" || $7 !~ /^context=[0-9]+$/ ||
        $8 != "splits=1" || $9 != "kv_type=e4m3" || $10 != "k_layout=dim-major" ||
        $11 != "v_layout=token-major" || $12 != "q_heads=6" || $13 != "kv_heads=1" ||
        $14 != "head_dim=256" || $15 !~ /^fused_ns=[1-9][0-9]*$/ || $16 != "reduce_ns=0") {
        fail("bad verifier attention event"); next
    }
    layer = substr($5, 7) + 0
    pos = substr($7, 9) + 0
    if (!layers) context = pos
    if (await_head || pos != context || layer >= 48 || seen[layer]++) fail("round/layer mismatch")
    ++layers
    ns += substr($15, 10) + 0
}
/^  qwen4_exp prefill M=4 \(sub-chunks of 4\): / {
    if (await_head || layers != 12 || (rounds && context <= previous) ||
        $13 != "full-attn" || $14 !~ /^[0-9]+\.[0-9]+$/ || $14 + 0 <= 0) {
        fail("incomplete four-row attention phase"); next
    }
    wall = $14 + 0
    if (is_split)
        printf "%d\t%d\t%d\t%.6f\t%.6f\t%.3f\t%.6f\n", rank, context, layers, ns / 1e6, reduction / 1e6, wall, wall - (ns + reduction) / 1e6
    else
        printf "%d\t%d\t%d\t%.6f\t%.3f\t%.6f\n", rank, context, layers, ns / 1e6, wall, wall - ns / 1e6
    previous = context
    ++rounds
    delete seen
    layers = ns = reduction = 0
    await_head = 1
}
/^QFN MTP3 verify head: / {
    if (!await_head || $5 != "rank=" rank || $6 != "pos=" context || $7 != "rows=4")
        fail("head belongs to a different round")
    await_head = 0
    ++heads
}
/^generation phases: / { fail("throughput receipt in a profile-only run") }
END { if (bad || !rounds || layers || await_head || rounds != heads) exit 1 }
