#include "b70/qfn_moe_routes.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using b70::make_qfn_moe_route_plan;
using b70::QfnMoeRoutePlan;

// Independent oracle: filter the original table once per expert instead of
// counting, prefix summing and scattering by the estimated offsets.
void check_plan(const QfnMoeRoutePlan& plan, const std::vector<std::int32_t>& ids,
                int top_k, int experts) {
    std::vector<std::int32_t> counts, offsets{0}, perm_token, inverse(ids.size(), -1);
    for (int e = 0; e < experts; ++e) {
        int count = 0;
        for (std::size_t slot = 0; slot < ids.size(); ++slot) {
            if (ids[slot] != e) continue;
            inverse[slot] = static_cast<std::int32_t>(perm_token.size());
            perm_token.push_back(static_cast<std::int32_t>(slot / top_k));
            ++count;
        }
        counts.push_back(count);
        offsets.push_back(static_cast<std::int32_t>(perm_token.size()));
    }
    assert(plan.counts == counts);
    assert(plan.offsets == offsets);
    assert(plan.perm_token == perm_token);
    assert(plan.inverse == inverse);
}

// Test the consumer contract independently of the index oracle: execute a toy
// expert per compacted row, then compare a weighted gather to a direct sum that
// never consults the plan. Padding has a NaN weight so it cannot be accumulated.
std::vector<float> check_gather(const QfnMoeRoutePlan& plan,
                                const std::vector<std::int32_t>& ids, int top_k, int rows) {
    constexpr int width = 3;
    std::vector<float> activations(rows * width);
    for (int i = 0; i < rows * width; ++i) activations[i] = float(i + 1);
    std::vector<float> weights(ids.size());
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
        weights[slot] = ids[slot] == -1 ? std::numeric_limits<float>::quiet_NaN()
                                       : float(slot + 1) / 64.0f;
    }
    std::vector<float> grouped(plan.perm_token.size() * width);
    for (std::size_t e = 0; e < plan.counts.size(); ++e) {
        for (int i = 0; i < plan.counts[e]; ++i) {
            const int p = plan.offsets[e] + i;
            const int token = plan.perm_token[p];
            for (int d = 0; d < width; ++d) {
                grouped[p * width + d] = activations[token * width + d] + float(e + 1);
            }
        }
    }
    std::vector<float> gathered(rows * width);
    for (int token = 0; token < rows; ++token) {
        for (int d = 0; d < width; ++d) {
            float expected = 0.0f;
            float actual = 0.0f;
            for (int s = 0; s < top_k; ++s) {
                const int slot = token * top_k + s;
                if (ids[slot] != -1) {
                    expected += weights[slot] * (activations[token * width + d] + float(ids[slot] + 1));
                }
                const int p = plan.inverse[slot];
                if (p >= 0) actual = std::fma(weights[slot], grouped[p * width + d], actual);
            }
            assert(actual == expected); // Small dyadic weights: both sums are exact.
            gathered[token * width + d] = actual;
        }
    }
    return gathered;
}

void test_flash_next() {
    constexpr int rows = 5, top_k = 10, experts = 512;
    const std::vector<std::int32_t> ids{
        511,  -1, 511,  17, 511, 511, 511,  -1, 510, 511,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  17, 511,
         -1, 511,   0, 511,  17,  -1, 511,  17, 511, 510,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         17, 511, 511, 511, 511, 511, 511, 511,  -1, 511,
    };
    const auto plan = make_qfn_moe_route_plan(ids.data(), ids.size(), rows, top_k, experts);
    check_plan(plan, ids, top_k, experts);
    const auto gathered = check_gather(plan, ids, top_k, rows);
    assert(plan.counts[0] == 1); // Expert zero is real; none of the padding belongs to it.
    assert(plan.inverse[18] >= 0 && plan.inverse[19] >= 0);
    assert(gathered[3] == (19.0f * 22.0f + 20.0f * 516.0f) / 64.0f); // Row 1: only slots 8 and 9.
    assert(gathered[9] == 0.0f); // Row 3: all padding in a nonempty group.
}

void test_grouped_tiles() {
    constexpr int rows = 80, top_k = 10, experts = 512;
    std::vector<int32_t> ids(rows * top_k, -1);
    for (int t = 0; t < 65; ++t) ids[t * top_k + 9] = 511;
    for (int t = 0; t < 32; ++t) ids[t * top_k + 8] = 0;
    ids[7] = 64;
    const auto plan = make_qfn_moe_route_plan(ids.data(), ids.size(), rows, top_k, experts);
    const auto tiles = b70::make_qfn_moe_tiles(plan);
    assert(tiles.size() == 5);
    const int32_t expected[][3]{{0,0,32}, {64,32,1}, {511,33,32}, {511,65,32}, {511,97,1}};
    for (size_t i = 0; i < tiles.size(); ++i) {
        assert(tiles[i].expert == expected[i][0]);
        assert(tiles[i].first_row == expected[i][1]);
        assert(tiles[i].rows == expected[i][2]);
    }
    assert(b70::make_qfn_moe_tiles(make_qfn_moe_route_plan(nullptr, 0, 0, 10, 512)).empty());
}

void test_empty_and_non_top10() {
    const auto zero_rows = make_qfn_moe_route_plan(nullptr, 0, 0, 10, 512);
    check_plan(zero_rows, {}, 10, 512);

    const std::vector<std::int32_t> padding(6, -1);
    const auto all_absent = make_qfn_moe_route_plan(padding.data(), padding.size(), 2, 3, 4);
    check_plan(all_absent, padding, 3, 4);
    check_gather(all_absent, padding, 3, 2);

    const std::vector<std::int32_t> dense{3, 0, 1, 1, 3, 2};
    const auto plan = make_qfn_moe_route_plan(dense.data(), dense.size(), 2, 3, 4);
    check_plan(plan, dense, 3, 4);
    check_gather(plan, dense, 3, 2);
}

template <class Error, class Call>
void expect_error(Call call) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    assert(false && "expected route plan construction to reject the input");
}

void test_tp4_resident_windows() {
    const std::vector<std::int32_t> global{0, 127, 128, 255, 256, 383, 384, 511, 128, 0, 255, -1};
    const auto original = global;
    for (int rank = 0; rank < 4; ++rank) {
        const int begin = rank * 128;
        auto local = global;
        for (auto& id : local) id = id >= begin && id < begin+128 ? id-begin : -1;
        const auto actual = make_qfn_moe_route_plan(global.data(), global.size(), 3, 4, 512, begin, 128);
        check_plan(actual, local, 4, 128);
        check_gather(actual, local, 4, 3);
        const auto expected = make_qfn_moe_route_plan(local.data(), local.size(), 3, 4, 128);
        assert(actual.inverse == expected.inverse && actual.perm_token == expected.perm_token);
        assert(global == original);
    }
    for (const int begin : {-1, 385, std::numeric_limits<int>::max()})
        expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(global.data(), global.size(), 3, 4, 512, begin, 128); });
    const std::int32_t bad[]{512};
    expect_error<std::out_of_range>([&] { make_qfn_moe_route_plan(bad, 1, 1, 1, 512, 0, 128); });
}

void test_bad_inputs() {
    std::int32_t ids[]{0, 0};
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 2, -1, 2, 512); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 0, 1, 0, 512); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 0, 1, -1, 512); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 2, 1, 2, 0); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 2, 1, 2, -1); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 1, 1, 2, 512); });
    expect_error<std::invalid_argument>([&] { make_qfn_moe_route_plan(ids, 2, 1, 1, 512); });
    expect_error<std::invalid_argument>([] { make_qfn_moe_route_plan(nullptr, 2, 1, 2, 512); });

    constexpr auto max = std::numeric_limits<std::int32_t>::max();
    expect_error<std::length_error>([] { make_qfn_moe_route_plan(nullptr, 0, max / 10 + 1, 10, 512); });
    expect_error<std::length_error>([] { make_qfn_moe_route_plan(nullptr, 0, max, max, 512); });
    expect_error<std::invalid_argument>([] { make_qfn_moe_route_plan(nullptr, 0, max, 1, 512); });
    for (const auto bad : {-2, std::numeric_limits<std::int32_t>::min(), 512, max}) {
        ids[1] = bad;
        expect_error<std::out_of_range>([&] { make_qfn_moe_route_plan(ids, 2, 1, 2, 512); });
    }
}

} // namespace

int main() {
    test_flash_next();
    test_grouped_tiles();
    test_empty_and_non_top10();
    test_tp4_resident_windows();
    test_bad_inputs();
    std::puts("QFN MoE route/gather regression passed");
}
