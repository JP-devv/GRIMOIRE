// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "weights.hpp"
#include <cstdint>

namespace b70 {

struct QfnInt4Pair {
    float lo;
    float hi;
};

// The two consecutive K elements in a byte share a scale and zero point:
// all supported INT4 groups have even boundaries. 0xff selects signed s4.
inline QfnInt4Pair qfn_decode_int4_pair(uint8_t byte, float scale, uint8_t zero) {
    const int flip = zero == 0xff ? 8 : 0;
    const int bias = zero == 0xff ? 8 : int(zero);
    return {float(((byte & 15) ^ flip) - bias) * scale,
            float(((byte >> 4) ^ flip) - bias) * scale};
}

inline QfnInt4Pair qfn_load_int4_pair(const QuantWeight& w, int64_t row, int even_k) {
    const int64_t group = row * w.row_scales + (even_k >> w.int4_gshift());
    const uint8_t packed = w.payload[row * w.row_bytes + even_k / 2];
    const float scale = bf16_to_f32(static_cast<const bf16_t*>(w.scales)[group]);
    return qfn_decode_int4_pair(packed, scale, w.zeros[group]);
}

// BF16 XMX B uses VNNI order: each adjacent pair along K is contiguous.
inline constexpr int qfn_int4_pair_vnni_offset(int even_k, int n, int columns) {
    return even_k * columns + 2 * n;
}

} // namespace b70
