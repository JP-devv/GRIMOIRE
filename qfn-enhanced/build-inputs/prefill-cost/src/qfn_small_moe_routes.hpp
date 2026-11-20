#pragma once
#include <cstdint>

namespace b70 {
inline constexpr int kSmallMoeRows = 4;
inline constexpr int kSmallMoeTopK = 10;
inline constexpr int kSmallMoeSlots = kSmallMoeRows * kSmallMoeTopK;
inline constexpr int kSmallMoeExperts = 128;
inline constexpr int kSmallMoeHidden = 2560;
inline constexpr int kSmallMoeInner = 640;

struct SmallMoeBatch { int32_t expert, begin, count; };
struct SmallMoeRoutes {
    int32_t batches;
    SmallMoeBatch batch[kSmallMoeSlots];
    int32_t slot[kSmallMoeSlots];    // compact -> original top-k slot
    int32_t inverse[kSmallMoeSlots]; // original -> compact, -1 for inactive
};

// The caller supplies 1–4 rows. Never read unused rows or visit nonlocal
// weights. Group by first occurrence and cap each expert batch at the input row count.
inline void plan_small_moe_routes(const int32_t* ids, const float* weights, int rows,
                                  int expert_begin, SmallMoeRoutes& plan) {
    const int slots = rows * kSmallMoeTopK;
    plan.batches = 0;
    int live = 0;
    for (int s = 0; s < slots; ++s) plan.inverse[s] = -1;
    for (int s = 0; s < slots; ++s) {
        const int id = ids[s];
        if (id < expert_begin || id >= expert_begin + kSmallMoeExperts ||
            weights[s] == 0.0f || plan.inverse[s] >= 0) continue;
        auto& batch = plan.batch[plan.batches++];
        batch = {id - expert_begin, live, 0};
        for (int j = s; j < slots && batch.count < rows; ++j) {
            if (ids[j] != id || weights[j] == 0.0f || plan.inverse[j] >= 0) continue;
            plan.slot[live] = j;
            plan.inverse[j] = live++;
            ++batch.count;
        }
    }
}

template<class Value, class Fma>
inline float fold_small_moe_routes(const SmallMoeRoutes& plan, const float* weights,
                                   int row, Value value, Fma fma) {
    float total = 0.0f;
    for (int i = 0; i < kSmallMoeTopK; ++i) {
        const int slot = row * kSmallMoeTopK + i;
        const int route = plan.inverse[slot];
        if (route >= 0) total = fma(weights[slot], value(route), total);
    }
    return total;
}
} // namespace b70
