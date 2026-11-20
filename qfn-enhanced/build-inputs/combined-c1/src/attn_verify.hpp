// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Speculative-verify attention on the matrix engine (src/attn_verify_dpas.cpp, built
// into bin/libgrimoire_attn.so with 256 registers per ESIMD thread).
#pragma once
#include "kernels.hpp"

namespace b70 {

// One group = the verify rows of one conversation: rows row0 .. row0+nrows-1, one KV
// slot, consecutive positions (row r has length d_lens[r]; the last row is the longest).
constexpr int kMaxVerifyGroups = 16;
// Attention partials are sized for kVerifyPartRows rows x MAX_SPLITS (grimoire.cpp s.part); verify_dpas_splits
// keeps rows x splits within that (a 33..64-row copy verify takes at most MAX_SPLITS / 2 splits).  11-15.
constexpr int kVerifyPartRows = 32;
struct VerifyGroups {
    int32_t n = 0;
    int32_t row0[kMaxVerifyGroups];
    int32_t nrows[kMaxVerifyGroups];
    int32_t slot[kMaxVerifyGroups];
};

// head_dim 256, FP8 E4M3 cache, no window / sparse bits, seq_cap a multiple of 16.
bool verify_dpas_ok(const AttnParams& p);
// Split count (the launch's splits_max) for these groups, len_max = the longest row's
// length over all of them: ~GRIMOIRE_VERIFY_THREADS threads, >= 32 keys per split.
int verify_dpas_splits(const AttnParams& p, const VerifyGroups& g, int len_max);
// p: row 0's q / out (row stride num_heads * head_dim), k/v_cache = the slot-0 bases,
// partials / part_m / part_l with room for rows * num_heads * splits_max.
// splits_max >= verify_dpas_splits(len_max) of every group.
// len_hint: the longest row's length (host-known at the call); 0 = unknown -> never the team kernel.
sycl::event launch_verify_attn_dpas(sycl::queue& q, const AttnParams& p, const VerifyGroups& g,
                                    int64_t kv_stride, const int32_t* d_lens, int splits_max,
                                    const std::vector<sycl::event>& deps = {}, int len_hint = 0);
// KEY-MAJOR K CACHE (2026-11-14): when set, K is stored like V, [kv_head][seq_cap][head_dim] bytes, and the
// verify kernel reads it with transposed 2-D block loads.  Set once at load (Grimoire::build, kv_kmajor_flag()) --
// this library keeps its own copy of the flag.
void set_verify_kv_kmajor(bool on);
void set_verify_kv_k8(bool on);       // int8 keys (kernels.hpp kv_k8_flag), key-major only
bool verify_kv_kmajor();
// TEAM verify attention (2026-11-15, key-major cache only): one work-group of 8 threads per (conversation, KV head,
// split) holds ALL of that KV head's verify rows (<= 48 = 8 tokens x 6 heads); thread t computes S for its own 16 of
// every 128-key block, the team shares the block max and P through SLM, then thread t accumulates O for its own 32 of
// the 256 dims.  Each K / V byte is loaded and widened ONCE (the per-tile kernel: once per 8-row tile, 6x for one
// user's 7 rows).  -1 = GRIMOIRE_VERIFY_TEAM (default on), 0 off, 1 on.  Used for >= 4 rows per conversation at
// len_hint >= GRIMOIRE_VERIFY_TEAM_MIN (32768).  MEASURED 2026-11-15 (tools/bench_verify_team, one B70, us per layer,
// per-tile -> team): 61K 1x7 577 -> 403, 2x7 991 -> 793, 4x7 1954 -> 1467, 4x4 1288 -> 1196, 8x4 2325 -> 2305;
// 32K 4x7 1131 -> 798; slower for 2-row groups (61K 1x2 254 -> 320) and short contexts (8K 1x7 74 -> 88).
void set_verify_team(int mode);
bool verify_team_on();
// out[row][head] = the merge of that row's splits.  Empty splits (m = -inf) are skipped.
sycl::event launch_verify_merge(sycl::queue& q, const AttnParams& p, int rows, int splits_max,
                                const std::vector<sycl::event>& deps = {});

} // namespace b70
