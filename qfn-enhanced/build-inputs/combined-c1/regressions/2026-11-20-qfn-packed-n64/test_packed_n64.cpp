// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_int4_pair.hpp"
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    for (unsigned b = 0; b < 256; ++b) {
        const auto payload = static_cast<uint8_t>(b);
        for (unsigned z = 0; z < 16 + 1; ++z) {
            const auto zero = uint8_t(z == 16 ? 0xff : z);
            for (float scale : {0.0f, 0.00390625f, 0.1435546875f, 3.125f}) {
                const auto pair = b70::qfn_decode_int4_pair(payload, scale, zero);
                assert(std::bit_cast<uint32_t>(pair.lo) ==
                       std::bit_cast<uint32_t>(b70::decode_int4(&payload, 0, scale, zero)));
                assert(std::bit_cast<uint32_t>(pair.hi) ==
                       std::bit_cast<uint32_t>(b70::decode_int4(&payload, 1, scale, zero)));
            }
        }
    }

    // All supported scale layouts, including row and group boundaries.
    // Give each group a different scale and alternate signed and affine.
    constexpr int outputs = 3, inner = 512;
    std::array<uint8_t, outputs * inner / 2> weights{};
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = uint8_t(i * 37 + 177);
    for (int group_width : {32, 64, 128, 256}) {
        const int groups = inner / group_width;
        std::vector<b70::bf16_t> scales(outputs * groups);
        std::vector<uint8_t> zeros(outputs * groups);
        for (size_t i = 0; i < scales.size(); ++i) {
            scales[i] = b70::f32_to_bf16(float(i + 1) / 256.0f);
            zeros[i] = i % 2 == 0 ? 0xff : uint8_t(i % 16);
        }
        const b70::QuantWeight w{b70::Fmt::INT4, outputs, inner, weights.data(),
                                 scales.data(), zeros.data(), inner / 2, groups};
        for (int row = 0; row < outputs; ++row) {
            for (int k = 0; k < inner; k += 2) {
                const auto pair = b70::qfn_load_int4_pair(w, row, k);
                assert(std::bit_cast<uint32_t>(pair.lo) == std::bit_cast<uint32_t>(w.at(row, k)));
                assert(std::bit_cast<uint32_t>(pair.hi) == std::bit_cast<uint32_t>(w.at(row, k + 1)));
            }
        }
    }

    // Check the packed B placement over a complete N64/K32 tile: no
    // overlaps or gaps between work-items' pair stores.
    constexpr int N = 64, K = 32;
    std::array<int, N * K> ownership{};
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const int dst = b70::qfn_int4_pair_vnni_offset(k, n, N);
            assert(dst == (k / 2) * N * 2 + n * 2);
            assert(dst >= 0 && dst + 1 < N * K);
            ++ownership[dst];
            ++ownership[dst + 1];
        }
    }
    for (int count : ownership) assert(count == 1);
    std::puts("QFN grouped INT4 N64: signed/affine pair decode and packed tile coverage PASS");
}
