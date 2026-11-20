#pragma once
#include "b70/moe.hpp"
#include "qfn_small_moe_routes.hpp"

namespace b70 {
inline bool small_moe_weights_ok(const MoeLayer& moe) {
    const auto packed = [](const QuantWeight& w, int n, int k) {
        return w.fmt == Fmt::INT4 && w.N == n && w.K == k && w.payload && w.scales && w.zeros &&
               w.int4_gshift() == 7 && w.row_bytes == k / 2 && w.row_scales == k / 128;
    };
    return moe.cfg.hidden == kSmallMoeHidden && moe.cfg.inter == kSmallMoeInner &&
           moe.cfg.top_k == kSmallMoeTopK && moe.cfg.num_experts == kSmallMoeExperts &&
           packed(moe.gate_up, kSmallMoeExperts * 2 * kSmallMoeInner, kSmallMoeHidden) &&
           packed(moe.down, kSmallMoeExperts * kSmallMoeHidden, kSmallMoeInner);
}

// Requires an in-order queue, qualified weights, and rows in [1, 4].
// h and values each have rows * kSmallMoeTopK rows of inner/hidden floats.
void launch_qfn_small_moe(sycl::queue& q, const MoeLayer& moe, int rows, int expert_begin,
                          const int32_t* ids, const float* weights, const float* x,
                          SmallMoeRoutes* plan, float* h, float* values, float* y);
} // namespace b70
