// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_longattn -- verify / draft attention at LONG context (2026-11-14), Qwen3.8-27B geometry (24 query heads, 4 KV
// heads, head_dim 256, FP8 E4M3 cache), one layer: launch_verify_attn_dpas + launch_verify_merge, each group its own
// cache slot (separate users).  Cases: (groups x rows) = 1x1 (one MTP draft row), 1x7 (one user's verify, MTP K=6),
// 1x16, 4x4 (four users' verify), 4x8, 8x2.  Prints us per layer and the K/V bytes read per us (each group reads
// its own L keys once: L x 4 heads x 256 x 2 bytes).  Correctness: 1x1 / 1x7 against a double reference at <= 16K.
//   bench_longattn [L ...]   (default 8192 32768 61440)
// Build (grimoire-dev, repo root, after bin/libgrimoire_attn.so):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src tools/bench_longattn.cpp \
//        -Lbin -lgrimoire_attn '-Wl,-rpath,$ORIGIN' -o bin/bench_longattn
#include <sycl/sycl.hpp>
#include "kernels.hpp"
#include "attn_verify.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace b70;

int main(int argc, char** argv) {
    const int H = 24, KVH = 4, HD = 256;
    std::vector<int> Ls;
    for (int i = 1; i < argc; ++i) Ls.push_back(std::atoi(argv[i]));
    if (Ls.empty()) Ls = {8192, 32768, 61440};
    int CAP = 16;
    for (int L : Ls) CAP = std::max(CAP, (L + 15) / 16 * 16);
    const int SLOTS = 8, RMAX = 32;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s  CAP %d  VERIFY_THREADS=%s KEYS_PER_SPLIT=%s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), CAP,
                std::getenv("GRIMOIRE_VERIFY_THREADS") ? std::getenv("GRIMOIRE_VERIFY_THREADS") : "-",
                std::getenv("GRIMOIRE_VERIFY_KEYS_PER_SPLIT") ? std::getenv("GRIMOIRE_VERIFY_KEYS_PER_SPLIT") : "-");
    const int64_t slot_bytes = int64_t(KVH) * HD * CAP;
    uint8_t* dk = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    uint8_t* dv = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    // random E4M3 bytes in a sane range: sign | exponent 4..9 | mantissa
    {
        std::mt19937 rng(7);
        std::vector<uint8_t> h(size_t(slot_bytes));
        for (auto& b : h) b = uint8_t((rng() & 0x80) | ((4 + rng() % 6) << 3) | (rng() & 7));
        for (int s = 0; s < SLOTS; ++s) {
            q.memcpy(dk + size_t(s) * slot_bytes, h.data(), h.size());
            q.memcpy(dv + size_t(s) * slot_bytes, h.data(), h.size());
        }
        q.wait();
    }
    std::vector<float> hq(size_t(RMAX) * H * HD);
    { std::mt19937 rng(3); std::normal_distribution<float> nd; for (auto& x : hq) x = nd(rng); }
    float* dq = sycl::malloc_device<float>(hq.size(), q);
    float* dout = sycl::malloc_device<float>(hq.size(), q);
    const size_t pcount = size_t(RMAX) * H * MAX_SPLITS;
    float* dpart = sycl::malloc_device<float>(pcount * HD, q);
    float* dpm = sycl::malloc_device<float>(pcount, q);
    float* dpl = sycl::malloc_device<float>(pcount, q);
    int32_t* dlens = sycl::malloc_device<int32_t>(RMAX, q);
    q.memcpy(dq, hq.data(), hq.size() * 4).wait();
    AttnParams base{};
    base.q = dq; base.k_cache = dk; base.v_cache = dv; base.out = dout;
    base.seq_len = 1; base.seq_cap = CAP; base.head_dim = HD; base.num_heads = H; base.num_kv_heads = KVH;
    base.softmax_scale = 1.0f / 16.0f; base.partials = dpart; base.part_m = dpm; base.part_l = dpl; base.splits = 8;
    for (int i = 0; i < 100; ++i) q.memset(dpart, 0, pcount * HD * 4);
    q.wait();
    struct Case { int groups, rows; };
    const Case cases[] = {{1, 1}, {1, 7}, {1, 16}, {1, 32}, {4, 1}, {4, 4}, {4, 8}, {8, 2}};
    std::printf("%7s %6s | %9s %8s %6s\n", "keys", "case", "us/layer", "GB/s", "splits");
    for (int L : Ls) {
        for (const Case c : cases) {
            if (c.groups * c.rows > RMAX) continue;
            VerifyGroups vg{};
            vg.n = c.groups;
            std::vector<int32_t> lens(size_t(c.groups) * c.rows);
            for (int g = 0; g < c.groups; ++g) {
                vg.row0[g] = g * c.rows; vg.nrows[g] = c.rows; vg.slot[g] = g;
                for (int r = 0; r < c.rows; ++r) lens[size_t(g) * c.rows + r] = L - c.rows + 1 + r;
            }
            q.memcpy(dlens, lens.data(), lens.size() * 4).wait();
            const int smax = verify_dpas_splits(base, vg, L);
            auto run = [&] {
                launch_verify_attn_dpas(q, base, vg, slot_bytes, dlens, smax);
                launch_verify_merge(q, base, c.groups * c.rows, smax);
            };
            for (int i = 0; i < 3; ++i) run();
            q.wait();
            const int N = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) run();
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            const double bytes = double(c.groups) * L * KVH * HD * 2;
            std::printf("%7d %3dx%-2d | %9.1f %8.1f %6d\n", L, c.groups, c.rows, us, bytes / us / 1e3, smax);
        }
    }
    return 0;
}
