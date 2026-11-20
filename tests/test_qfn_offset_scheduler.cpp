// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_moe_offset_schedule.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    // The descriptor scheduler is the oracle. Include no work, sparse
    // expert spans, 31/32/33 rows, and more live N64 tiles than workers.
    const std::vector<std::vector<int>> shapes{
        {0, 0, 0}, {0, 1, 0, 31, 32, 0, 33, 0, 64, 65, 0},
        std::vector<int>(128, 40), {32, 32, 32, 32}
    };
    for (const auto& counts : shapes) {
        b70::QfnMoeRoutePlan plan;
        plan.counts.assign(counts.begin(), counts.end());
        plan.offsets.resize(counts.size() + 1);
        for (size_t e = 0; e < counts.size(); ++e)
            plan.offsets[e + 1] = plan.offsets[e] + counts[e];
        plan.perm_token.resize(plan.offsets.back());
        const auto reference = b70::make_qfn_moe_tiles(plan);
        for (int ntiles : {1, 20}) for (int workers : {1, 7, 256}) {
            const int total = int(reference.size()) * ntiles;
            std::vector<int> seen(total);
            for (int g = 0; g < workers; ++g) {
                b70::QfnMoeOffsetCursor cursor;
                for (int work = g;; work += workers) {
                    const auto tile = cursor.find(work / ntiles, plan.offsets.data(), int(counts.size()));
                    if (work >= total) {
                        assert(tile.expert == -1 && tile.rows == 0);
                        break;
                    }
                    const auto expected = reference[work / ntiles];
                    assert(tile.expert == expected.expert && tile.first_row == expected.first_row && tile.rows == expected.rows);
                    ++seen[work];
                }
            }
            assert(std::all_of(seen.begin(), seen.end(), [](int count) { return count == 1; }));
        }
    }
    std::cout << "QFN offset scheduler: exact N64 coverage, empty experts and ragged rows PASS\n";
}
