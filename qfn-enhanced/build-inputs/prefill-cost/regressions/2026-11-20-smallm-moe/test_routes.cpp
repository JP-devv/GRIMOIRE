#include "qfn_small_moe_routes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace b70;

static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static constexpr auto weighted_add = [](float w, float v, float sum) { return std::fma(w, v, sum); };

int main() {
    SmallMoeRoutes plan{};
    // A, B, A in row 0 is intentionally order-sensitive. Group execution
    // visits A, A, B, while scatter must sum +large, -large, +1.
    std::array<int32_t, kSmallMoeSlots> ids;
    std::array<float, kSmallMoeSlots> weights;
    ids.fill(-1);
    weights.fill(1.0f);
    ids[0] = 129;
    ids[1] = 130;
    ids[2] = 129;
    ids[3] = 131;
    weights[3] = -0.0f;
    ids[4] = 0;
    ids[5] = 255;
    ids[6] = 256;
    ids[10] = 130;
    ids[25] = 129;
    ids[34] = 345;
    constexpr std::array<float, kSmallMoeSlots> values{1e20f, -1e20f, 1.0f};
    for (int rows = 1; rows <= kSmallMoeRows; ++rows) {
        const int slots = rows * kSmallMoeTopK;
        // Exact-length buffers (no padded rows available to the planner).
        const std::vector<int32_t> row_ids(ids.begin(), ids.begin() + slots);
        const std::vector<float> row_weights(weights.begin(), weights.begin() + slots);
        for (int begin = 0; begin < 4 * kSmallMoeExperts; begin += kSmallMoeExperts) {
            std::array<bool, kSmallMoeSlots> visited{};
            std::array<int, kSmallMoeSlots> read_order{};
            plan_small_moe_routes(row_ids.data(), row_weights.data(), rows, begin, plan);
            int live = 0, prior_first = -1;
            for (int i = 0; i < plan.batches; ++i) {
                const auto batch = plan.batch[i];
                require(batch.begin == live && batch.count > 0 && batch.count <= rows, "bad compact batch");
                require(plan.slot[live] > prior_first, "batch order not based on first occurrence");
                int prior = prior_first = plan.slot[live];
                for (int j = 0; j < batch.count; ++j, ++live) {
                    const int s = plan.slot[live];
                    require(s >= 0 && s < slots && (j == 0 || s > prior), "bad slot order in expert batch");
                    require(!visited[s] && plan.inverse[s] == live && ids[s] == begin + batch.expert,
                            "bad inverse, expert, or duplicate route");
                    visited[s] = true;
                    prior = s;
                }
            }
            for (int row = 0; row < rows; ++row) {
                float ref = 0.0f;
                std::vector<int> expected;
                for (int s = row * kSmallMoeTopK; s < (row + 1) * kSmallMoeTopK; ++s) {
                    const bool active = ids[s] >= begin && ids[s] < begin + kSmallMoeExperts && weights[s] != 0.0f;
                    require(visited[s] == active && (plan.inverse[s] >= 0) == active, "lost live route or read inactive slot");
                    if (active) {
                        expected.push_back(s);
                        ref = weighted_add(weights[s], values[s], ref);
                    }
                }
                int reads = 0;
                const float sum = fold_small_moe_routes(plan, row_weights.data(), row, [&](int route) {
                    require(route >= 0 && route < live, "read stale route value");
                    read_order[reads++] = plan.slot[route];
                    return values[plan.slot[route]];
                }, weighted_add);
                require(std::equal(expected.begin(), expected.end(), read_order.begin(), read_order.begin()+reads),
                        "scatter did not read original top-k order");
                require(sum == ref, "weighted scatter mismatch");
                if (begin == 128 && row == 0) require(sum == 1.0f, "cancellation did not preserve router order");
                if (begin == 384)
                    require(plan.batches == 0 && reads == 0 && sum == 0.0f && !std::signbit(sum),
                            "empty rank must contribute +0 without reading any values");
            }
        }
    }

    ids.fill(128);
    weights.fill(1.0f);
    plan_small_moe_routes(ids.data(), weights.data(), kSmallMoeRows, 128, plan);
    require(plan.batches == kSmallMoeSlots / kSmallMoeRows, "duplicate expert batch cap");
    for (int s = 0; s < kSmallMoeSlots; ++s) require(plan.inverse[s] == s, "lost duplicate slot");
    for (int s = 0; s < kSmallMoeSlots; ++s) ids[s] = 128 + s;
    plan_small_moe_routes(ids.data(), weights.data(), kSmallMoeRows, 128, plan);
    require(plan.batches == kSmallMoeSlots, "all-singleton capacity");
    std::puts("Small-M route ordering, short rows, capacity and empty ranks PASS");
}
