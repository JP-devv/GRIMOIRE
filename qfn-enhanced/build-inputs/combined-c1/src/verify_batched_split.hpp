#pragma once

namespace b70 {

struct VerifyBatchedSplit {
    int begin, end, row_end;  // Key counts; all ends are exclusive.
    // Tile iteration starts at begin, so only the two upper bounds matter.
    constexpr bool valid_key(int key) const { return key < end; }
    constexpr bool visible_key(int key) const { return valid_key(key) && key < row_end; }
    constexpr bool empty_for_row() const { return begin >= end || begin >= row_end; }
};

// base_seq_len already includes row zero's key. All rows share the last
// row's partition, then individually mask keys after their causal cutoff.
constexpr VerifyBatchedSplit verify_batched_split(int base_seq_len, int rows, int row, int part, int splits) {
    const int end = base_seq_len + rows - 1;
    const int step = (end + splits - 1) / splits;
    const int begin = part * step;
    const int next = begin + step;
    return {begin, next < end ? next : end, base_seq_len + row};
}

// Use a negative sentinel, not zero: a valid weight can underflow to zero.
constexpr float kVerifyEmptySplitWeight = -1.0f;

template<class Value, class Weight>
inline float verify_batched_weighted_sum(int splits, Value value, Weight weight) {
    float sum = 0.0f;
    for (int i = 0; i < splits; ++i) {
        const float w = weight(i);
        if (w != kVerifyEmptySplitWeight) sum += value(i) * w;
    }
    return sum;
}
} // namespace b70
