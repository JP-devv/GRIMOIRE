#include "b70/weights.hpp"

#include <array>
#include <cstddef>
#include <cstdio>

int main() {
    constexpr int group_size = 32;
    constexpr uint8_t offset_binary_zero = 8;
    constexpr uint8_t signed_s4 = 0xff;

    // Two little-endian I32 words: 0x76543210, 0xfedcba98. Every code
    // occurs exactly once; reversing nibbles or words must fail.
    constexpr std::array<uint8_t, 8> all_codes{
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
    };
    auto signed_codes = all_codes;
    for (auto& byte : signed_codes) byte ^= 0x88; // Flip the high bit of each nibble.
    for (int i = 0; i < 16; ++i) {
        const float expected = float(i - 8);
        if (b70::decode_int4(all_codes.data(), i, 1.0f, offset_binary_zero) != expected ||
            b70::decode_int4(all_codes.data(), i, 1.0f, signed_s4) == expected ||
            b70::decode_int4(signed_codes.data(), i, 1.0f, signed_s4) != expected) {
            std::fprintf(stderr, "all-code fixture failed at column %d (expected %g)\n", i, expected);
            return 1;
        }
    }

    // First two groups of expert 0 gate_proj, output row 0 in the MTP shard.
    constexpr std::array<uint8_t, 32> gate_payload{
        0x9d, 0x93, 0x66, 0x8b, 0x3b, 0x9d, 0xc8, 0x65,
        0x8a, 0x37, 0x74, 0x7d, 0xd8, 0x8a, 0xb7, 0x7f,
        0x79, 0x78, 0x76, 0x67, 0x54, 0x95, 0x81, 0x81,
        0x78, 0x74, 0xa9, 0x58, 0x57, 0x59, 0x98, 0x95,
    };
    constexpr std::array<int, 64> unpacked{
         5,  1, -5,  1, -2, -2,  3,  0,
         3, -5,  5,  1,  0,  4, -3, -2,
         2,  0, -1, -5, -4, -1,  5, -1,
         0,  5,  2,  0, -1,  3,  7, -1,
         1, -1,  0, -1, -2, -1, -1, -2,
        -4, -3, -3,  1, -7,  0, -7,  0,
         0, -1, -4, -1,  1,  2,  0, -3,
        -1, -3,  1, -3,  0,  1, -3,  1,
    };
    constexpr std::array<b70::bf16_t, 2> scales{{{0x3ba2}, {0x3c0b}}};
    constexpr std::array<uint8_t, 2> zeros{offset_binary_zero, offset_binary_zero};
    const b70::QuantWeight gate{b70::Fmt::INT4, 1, int(unpacked.size()), gate_payload.data(),
                               scales.data(), zeros.data(), int(gate_payload.size()), int(scales.size())};
    if ((1 << gate.int4_gshift()) != group_size) {
        std::fputs("draft gate fixture did not select group-32\n", stderr);
        return 1;
    }
    for (std::size_t k = 0; k < unpacked.size(); ++k) {
        const float scale = b70::bf16_to_f32(scales[k / group_size]);
        const float expected = float(unpacked[k]) * scale;
        const float actual = gate.at(0, int(k));
        if (actual != expected) {
            std::fprintf(stderr, "draft gate fixture failed at column %zu: expected %g, got %g\n",
                         k, expected, actual);
            return 1;
        }
    }
    std::puts("QFN draft INT4: all codes and two stored gate groups match with zero=8");
}
