#include "qfn_hc_down_schedule.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace b70;
constexpr int n = 320, k = 10240, baseline_m = 256, n_tile = kQfnHcDownTileN;
constexpr int padded = gemm_bf16_pitch(n, 256, false), packed = gemm_bf16_pitch(n, 256, true);
static_assert(padded == 512 && packed == 320);
static_assert(qfn_hc_down_m64_requested("1"));
static_assert(!qfn_hc_down_m64_requested(nullptr) && !qfn_hc_down_m64_requested("0") &&
              !qfn_hc_down_m64_requested("") && !qfn_hc_down_m64_requested("10"));

int main() {
    for (int m : {32, 63, 64, 65, 658, 2048, 2049}) {
        for (int group_m : {1, 2, 3, 4, 5, 8, 16}) {
            for (int tile_m : {baseline_m, kQfnHcDownTileM}) {
                const int mb = (m + tile_m - 1) / tile_m, nb = n / n_tile;
                std::vector<unsigned char> seen(m * nb, 0);
                for (int wg = 0; wg < mb * nb; ++wg) {
                    const auto tile = gemm_grouped_tile(wg, mb, nb, group_m);
                    assert(tile.m >= 0 && tile.m < mb && tile.n >= 0 && tile.n < nb);
                    if (tile_m == baseline_m) {
                        const int per = group_m * nb;
                        const int m0 = wg / per * group_m;
                        const int gm = mb - m0 < group_m ? mb - m0 : group_m;
                        assert(tile.m == m0 + (wg % per) % gm && tile.n == (wg % per) / gm);
                    }
                    // Production subgroups still own 32 activation rows;
                    // the final checked load/store may clip that footprint.
                    for (int sm = 0; sm < tile_m / 32; ++sm) {
                        const int start = tile.m * tile_m + sm * 32;
                        for (int row = start; row < start + 32 && row < m; ++row)
                            assert(++seen[row * nb + tile.n] == 1);
                    }
                }
                for (auto hits : seen) assert(hits == 1);
            }
        }
    }
    std::vector<uint16_t> compact(k * packed), old(k * padded, 0);
    for (int pair = 0; pair < k / 2; ++pair) for (int col = 0; col < n; ++col) {
        const auto a = gemm_vnni_pair_offset(pair, col, packed);
        const auto b = gemm_vnni_pair_offset(pair, col, padded);
        assert(a == std::size_t(2 * pair * n + 2 * col));
        assert(b == std::size_t(2 * pair * padded + 2 * col));
        const auto low = uint16_t(pair ^ col), high = uint16_t((pair + 1217) ^ col);
        compact.at(a) = old.at(b) = low;
        compact.at(a+1) = old.at(b+1) = high;
    }
    // Every N64 fragment and every K pair must read the same operands,
    // including the last fragment on the last packed row.
    for (int pair = 0; pair < k / 2; ++pair) for (int c = 0; c < n; c += n_tile) {
        const auto a = gemm_vnni_pair_offset(pair, c, packed);
        const auto b = gemm_vnni_pair_offset(pair, c, padded);
        for (int i = 0; i < 2 * n_tile; ++i) assert(compact.at(a+i) == old.at(b+i));
    }
    assert(gemm_vnni_pair_offset(k/2-1, n-1, packed) + 2 == compact.size());
    std::puts("QFN HC M64 CPU regression PASS (opt-in, grouped grid, partial tiles, compact BF16 pitch)");
}
