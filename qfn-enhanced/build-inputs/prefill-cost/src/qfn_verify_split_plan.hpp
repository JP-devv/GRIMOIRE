#pragma once

namespace b70 {

struct QfnVerifyAttention {
    bool split_requested, tp4_verifier, exact, dim_major_e4m3, in_order, has_partials;
    int rows, position, capacity, q_heads, kv_heads, head_dim;
};

inline constexpr int kQfnVerifySharedRows = 4;
inline constexpr int kQfnVerifySharedSplits = 32;

// Both callbacks inherit the caller's cache slot and dependency chain. The
// block kernel takes the zero-based anchor; batched decode takes the number
// of visible keys for row 0, including that row's own freshly appended key.
template<class Trace, class Batched, class Unsplit>
constexpr void dispatch_qfn_verify_attention(const QfnVerifyAttention& a,
                                              Trace trace, Batched batched, Unsplit unsplit) {
    const bool shared = a.split_requested && a.tp4_verifier && !a.exact &&
        a.dim_major_e4m3 && a.in_order && a.has_partials && a.rows == kQfnVerifySharedRows &&
        a.q_heads == 6 && a.kv_heads == 1 && a.head_dim == 256 &&
        a.capacity >= a.rows && a.position >= 2048 && a.position <= a.capacity - a.rows;
    trace(shared ? kQfnVerifySharedSplits : 1);
    if (shared) batched(a.position + 1, kQfnVerifySharedSplits);
    else unsplit(a.position);
}

} // namespace b70
