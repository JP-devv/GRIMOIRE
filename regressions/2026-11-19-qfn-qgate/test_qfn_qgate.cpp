// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_shard_plan.hpp"
#include "b70/qgate_layout.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr int heads = 24, dim = 256, tokens = 2, half = heads * dim, rows = 2 * half;

void require(bool ok, const char* test, int rank = -1, int index = -1) {
    if (ok) return;
    std::cerr << test << ": rank=" << rank << " index=" << index << '\n';
    std::exit(1);
}

int read_i32(std::istream& in) {
    unsigned char b[4];
    require(bool(in.read(reinterpret_cast<char*>(b), sizeof b)), "truncated vLLM golden");
    return int(uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24));
}
} // namespace

int main(int argc, char** argv) {
    using namespace b70;
    require(argc == 3, "usage: test_qfn_qgate VLLM_FORWARD_GOLDEN VLLM_LOADER_RANK_ROWS");
    std::ifstream fixture(argv[1], std::ios::binary), loader(argv[2], std::ios::binary);
    require(bool(fixture) && bool(loader), "open vLLM goldens");
    std::vector<int> packed, grouped, gold_q, gold_gate;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h)
            for (int x = 0; x < 2; ++x)
                for (int d = 0; d < dim; ++d)
                    packed.push_back(t * 1000000 + h * 10000 + x * 1000 + d);
        // Mutation: assume the checkpoint has one global Q half followed by
        // the gate half instead of one pair per head.
        for (int x = 0; x < 2; ++x)
            for (int h = 0; h < heads; ++h)
                for (int d = 0; d < dim; ++d)
                    grouped.push_back(t * 1000000 + h * 10000 + x * 1000 + d);
    }
    for (int i = 0; i < tokens * half; ++i) {
        gold_q.push_back(read_i32(fixture));
        gold_gate.push_back(read_i32(fixture));
        const auto p = qgate_offsets(i, dim); // production non-TP split
        require(packed.at(p.q) == gold_q.back() && packed.at(p.gate) == gold_gate.back(),
                "PP4 vs installed vLLM", -1, i);
    }
    require(fixture.peek() == std::char_traits<char>::eof(), "trailing golden bytes");

    QfnShardShape s;
    s.layers = 48; s.hidden = 2560;
    s.q_heads = heads; s.kv_heads = 2; s.attn_head_dim = dim; s.q_proj_rows = rows;
    s.gdn_k_heads = 16; s.gdn_v_heads = 48;
    s.gdn_k_head_dim = 128; s.gdn_v_head_dim = 128; s.gdn_conv_kernel = 4;
    s.routed_experts = 512; s.experts_per_token = 10; s.routed_width = 640; s.shared_width = 640;
    for (size_t l = 0; l < s.layer_kinds.size(); ++l)
        s.layer_kinds[l] = l % 4 == 3 ? QfnLayerKind::attn : QfnLayerKind::gdn;

    std::vector<int> reconstructed_q(tokens * half, -1), reconstructed_gate(tokens * half, -1);
    int end = 0;
    for (int r = 0; r < QfnShardPlan::world; ++r) {
        const char* refusal = nullptr;
        const auto plan = QfnShardPlan::make(s, QfnShardPlan::world, r, QfnSharedLayout::g128, refusal);
        require(bool(plan), refusal ? refusal : "construct production TP4 plan", r);
        const auto span = q_proj_head_rows(rows, heads, dim, int(plan->q_heads.begin), int(plan->q_heads.count));
        require(span.begin == end && span.count == 2 * dim * (heads / QfnShardPlan::world), "TP4 row coverage", r);
        end = span.begin + span.count;
        std::vector<int> local, mutated;
        for (int t = 0; t < tokens; ++t) {
            const int first = t * rows + span.begin;
            for (int i = 0; i < span.count; ++i) {
                const int selected = packed.at(first + i);
                require(selected == read_i32(loader), "raw TP4 slice vs installed vLLM loader", r, first + i);
                local.push_back(selected);
            }
            mutated.insert(mutated.end(), grouped.begin() + first, grouped.begin() + first + span.count);
        }
        bool rejected_global_halves = false, rejected_swapped_halves = false;
        for (int i = 0; i < tokens * span.count / 2; ++i) {
            const auto tp = qgate_offsets(i, dim);
            const int offset = (i / (span.count / 2)) * half + int(plan->q_heads.begin) * dim + i % (span.count / 2);
            const auto pp = qgate_offsets(offset, dim);
            require(local.at(tp.q) == packed.at(pp.q) && local.at(tp.gate) == packed.at(pp.gate),
                    "TP4 vs PP4", r, offset);
            require(reconstructed_q[offset] == -1 && reconstructed_gate[offset] == -1,
                    "rank output overlap", r, offset);
            reconstructed_q[offset] = local.at(tp.q);
            reconstructed_gate[offset] = local.at(tp.gate);
            rejected_global_halves |= mutated.at(tp.q) != gold_q[offset] || mutated.at(tp.gate) != gold_gate[offset];
            rejected_swapped_halves |= local.at(tp.gate) != gold_q[offset] || local.at(tp.q) != gold_gate[offset];
        }
        require(rejected_global_halves && rejected_swapped_halves, "mutation escaped detection", r);
        std::cout << "rank " << r << ": q_proj rows [" << span.begin << ',' << end
                  << "), raw slice matches vLLM loader and Q/gate match PP4; both mutations rejected\n";
    }
    require(end == rows, "rank spans do not cover full q_proj");
    require(loader.peek() == std::char_traits<char>::eof(), "trailing loader golden bytes");
    require(reconstructed_q == gold_q && reconstructed_gate == gold_gate,
            "reconstructed TP4 Q/gate output vs unsharded vLLM forward");
    std::cout << "reconstructed TP4: " << gold_q.size() << " Q/gate pairs match unsharded vLLM\n";
}
