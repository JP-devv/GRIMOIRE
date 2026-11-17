// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// INT8 KEYS (2026-11-17, kernels.hpp kv_k8_flag): the K cache writer.  One sub-group of 16 per (row, KV head): the
// head's dims are read once, s = max |k| / 127, k -> round(k / s) as int8 into the key-major cache, s into the scale
// area after the keys; V is written as FP8 E4M3 exactly as the FP8 writers do.  Rows whose position is outside the
// cache are skipped (the condition is uniform over the sub-group, so the reduction is never split).
#pragma once
#include "kernels.hpp"
#include "b70/formats.hpp"
#include <vector>

namespace b70 {

// T: float or sycl_bf16 rows [rows][kv_heads][head_dim].  PosF(r) -> position, SlotF(r) -> slot (cache at
// base + slot * kv_stride).  head_dim <= 256.
template <typename T, typename PosF, typename SlotF>
sycl::event kv_append_k8(sycl::queue& q, const T* k, const T* v, uint8_t* k_base, uint8_t* v_base,
                         int64_t kv_stride, PosF posf, SlotF slotf, int rows, int kv_heads, int head_dim,
                         int seq_cap, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * kv_heads * 16, 16),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int g = int(it.get_group(0)), lane = int(it.get_local_id(0));
                const int r = g / kv_heads, kh = g % kv_heads;
                const int pos = posf(r);
                if (pos < 0 || pos >= seq_cap) return;
                const int64_t slot = slotf(r);
                uint8_t* kc = k_base + slot * kv_stride;
                uint8_t* vc = v_base + slot * kv_stride;
                const int64_t src = (int64_t(r) * kv_heads + kh) * head_dim;
                float x[16];
                float mx = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = lane + 16 * i;
                    x[i] = d < head_dim ? float(k[src + d]) : 0.0f;
                    mx = sycl::fmax(mx, sycl::fabs(x[i]));
                }
                mx = sycl::reduce_over_group(it.get_sub_group(), mx, sycl::maximum<float>());
                const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
                uint8_t* kd = kc + (int64_t(kh) * seq_cap + pos) * head_dim;
                uint8_t* vd = vc + (int64_t(kh) * seq_cap + pos) * head_dim;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = lane + 16 * i;
                    if (d >= head_dim) break;
                    const float qv = sycl::clamp(sycl::rint(x[i] * inv), -127.0f, 127.0f);
                    kd[d] = uint8_t(int8_t(qv));
                    vd[d] = f32_to_e4m3(float(v[src + d]));
                }
                if (lane == 0)
                    reinterpret_cast<float*>(kc + int64_t(kv_heads) * head_dim * seq_cap)[int64_t(kh) * seq_cap + pos] =
                        mx / 127.0f;
            });
    });
}

} // namespace b70
