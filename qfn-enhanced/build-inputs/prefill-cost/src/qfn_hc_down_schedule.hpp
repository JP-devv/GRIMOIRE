#pragma once
#include <cstddef>

namespace b70 {
inline constexpr int kQfnHcDownTileM = 64;
inline constexpr int kQfnHcDownTileN = 64;

constexpr bool qfn_hc_down_m64_requested(const char* flag) {
    return flag && flag[0] == '1' && flag[1] == '\0';
}

constexpr int gemm_bf16_pitch(int n, int alignment, bool compact) {
    return compact ? n : ((n + alignment - 1) / alignment) * alignment;
}

constexpr std::size_t gemm_vnni_pair_offset(int pair, int n, int pitch) {
    return (std::size_t(pair) * pitch + n) * 2;
}

struct GemmGroupedTile { int m, n; };

// M runs fastest within packs of group_m workgroup rows. The last pack
// may be short; compact it without introducing holes or invalid N tiles.
constexpr GemmGroupedTile gemm_grouped_tile(int linear, int m_tiles, int n_tiles, int group_m) {
    const int pack_size = group_m * n_tiles;
    const int m_base = (linear / pack_size) * group_m;
    const int left = m_tiles - m_base;
    const int m_count = left < group_m ? left : group_m;
    const int within = linear % pack_size;
    return {m_base + within % m_count, within / m_count};
}
} // namespace b70
