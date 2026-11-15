// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_verify_team (2026-11-15) -- the TEAM verify attention (src/attn_verify_dpas.cpp, set_verify_team(1)) against the
// per-tile kernel it replaces (set_verify_team(0)): same key-major FP8 cache, same groups, same splits, both merged with
// launch_verify_merge; prints both times and the largest difference of the merged outputs (relative to the largest
// output).  Qwen3.8-27B geometry: 24 query heads, 4 KV heads, head_dim 256.
//   bench_verify_team [L ...]   (default 4096 8192 32768 61440)
// Build (grimoire-dev, repo root; the kernels need -doubleGRF):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -doubleGRF" \
//        -O3 -std=c++20 -I include -I src tools/bench_verify_team.cpp src/attn_verify_dpas.cpp -o bin/bench_verify_team
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
    if (Ls.empty()) Ls = {4096, 8192, 32768, 61440};
    int CAP = 16;
    for (int L : Ls) CAP = std::max(CAP, (L + 127) / 128 * 128);
    const int SLOTS = 8, RMAX = 32;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t slot_bytes = int64_t(KVH) * HD * CAP;
    uint8_t* dk = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    uint8_t* dv = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    {
        // finite E4M3 bytes (exponent 4..9), different K and V per slot
        std::mt19937 rng(7);
        std::vector<uint8_t> h(static_cast<size_t>(slot_bytes));
        for (int s = 0; s < SLOTS; ++s) {
            for (auto& b : h) b = uint8_t((rng() & 0x80) | ((4 + rng() % 6) << 3) | (rng() & 7));
            q.memcpy(dk + size_t(s) * slot_bytes, h.data(), h.size()).wait();
            for (auto& b : h) b = uint8_t((rng() & 0x80) | ((4 + rng() % 6) << 3) | (rng() & 7));
            q.memcpy(dv + size_t(s) * slot_bytes, h.data(), h.size()).wait();
        }
    }
    set_verify_kv_kmajor(true);
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
    struct Case { int groups, rows; };
    const Case cases[] = {{1, 7}, {1, 2}, {1, 8}, {2, 7}, {4, 4}, {4, 7}, {8, 2}, {8, 4}};
    std::printf("%7s %5s | %9s %9s %7s | %10s | %6s | GB/s tile / team\n", "keys", "case", "tile us", "team us",
                "speedup", "max rel d", "splits");
    for (int L : Ls)
        for (const Case c : cases) {
            if (c.groups * c.rows > RMAX) continue;
            VerifyGroups vg{};
            vg.n = c.groups;
            std::vector<int32_t> lens(size_t(c.groups) * c.rows);
            for (int g = 0; g < c.groups; ++g) {
                vg.row0[g] = g * c.rows; vg.nrows[g] = c.rows; vg.slot[g] = g;
                // conversations of slightly different lengths (the longest is L)
                const int Lg = L - 37 * g;
                for (int r = 0; r < c.rows; ++r) lens[size_t(g) * c.rows + r] = Lg - c.rows + 1 + r;
            }
            q.memcpy(dlens, lens.data(), lens.size() * 4).wait();
            const int smax = verify_dpas_splits(base, vg, L);
            const int rows = c.groups * c.rows;
            auto run = [&](int team) {
                set_verify_team(team);
                launch_verify_attn_dpas(q, base, vg, slot_bytes, dlens, smax, {});
                launch_verify_merge(q, base, rows, smax);
            };
            auto timeit = [&](int team) {
                set_verify_team(team);
                for (int i = 0; i < 3; ++i) launch_verify_attn_dpas(q, base, vg, slot_bytes, dlens, smax, {});
                q.wait();
                const int N = 20;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) launch_verify_attn_dpas(q, base, vg, slot_bytes, dlens, smax, {});
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            };
            const size_t nout = size_t(rows) * H * HD;
            std::vector<float> o0(nout), o1(nout);
            run(0); q.memcpy(o0.data(), dout, nout * 4).wait();
            q.memset(dout, 0xFF, nout * 4).wait();          // NaN: a row the team path forgets shows up
            run(1); q.memcpy(o1.data(), dout, nout * 4).wait();
            double md = 0, mx = 0; size_t bad = 0;
            for (size_t i = 0; i < nout; ++i) {
                if (!std::isfinite(o1[i])) { ++bad; continue; }
                md = std::max(md, double(std::fabs(o1[i] - o0[i])));
                mx = std::max(mx, double(std::fabs(o0[i])));
            }
            const double t0 = timeit(0), t1 = timeit(1);
            double bytes = 0;
            for (int g = 0; g < c.groups; ++g) bytes += 2.0 * KVH * HD * double(L - 37 * g);
            std::printf("%7d %2dx%-2d | %9.1f %9.1f %6.2fx | %10.3g | %6d | %6.0f / %6.0f%s\n", L, c.groups, c.rows, t0, t1,
                        t0 / t1, mx > 0 ? md / mx : md, smax, bytes / t0 / 1e3, bytes / t1 / 1e3,
                        bad ? "  NONFINITE ROWS" : "");
            std::fflush(stdout);
        }
    return 0;
}
