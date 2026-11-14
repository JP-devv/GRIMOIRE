// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_attn_bisect (2026-11-14) -- WHERE does the verify / draft attention kernel's time go at long context?
// Four builds of launch_verify_attn_dpas (tools/attn_bisect_kernels.inc = src/attn_verify_dpas.cpp + variants):
//   MODE 0 full | 1 no FP8->fp16 conversion (raw bytes) | 2 no DPAS (a cheap multiply-add) | 3 neither.
// Same grid, loads, splits and merge.  full - mode3 = conversion + matrix work; full - mode1 = conversion;
// full - mode2 = matrix work; mode3 = memory + softmax + the rest.  Qwen3.8-27B geometry, FP8 cache, one layer,
// groups x rows = 1x1 (one MTP draft row), 1x7 (one user's verify), 4x4 (four users).
// Build (grimoire-dev, repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -doubleGRF" \
//        -O3 -std=c++20 -I include -I src tools/bench_attn_bisect.cpp -o bin/bench_attn_bisect
#include "attn_bisect_kernels.inc"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace b70;

int main(int argc, char** argv) {
    const int H = 24, KVH = 4, HD = 256;
    std::vector<int> Ls;
    for (int i = 1; i < argc; ++i) Ls.push_back(std::atoi(argv[i]));
    if (Ls.empty()) Ls = {8192, 61440};
    int CAP = 16;
    for (int L : Ls) CAP = std::max(CAP, (L + 15) / 16 * 16);
    const int SLOTS = 4, RMAX = 16;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int64_t slot_bytes = int64_t(KVH) * HD * CAP;
    uint8_t* dk = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    uint8_t* dv = sycl::malloc_device<uint8_t>(size_t(slot_bytes) * SLOTS, q);
    {
        std::mt19937 rng(7);
        std::vector<uint8_t> h(static_cast<size_t>(slot_bytes));
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
    const Case cases[] = {{1, 1}, {1, 7}, {4, 4}};
    std::printf("%7s %5s | %9s %9s %9s %9s | GB/s full | splits\n", "keys", "case", "full", "noconv", "nodpas", "neither");
    for (int L : Ls)
        for (const Case c : cases) {
            VerifyGroups vg{};
            vg.n = c.groups;
            std::vector<int32_t> lens(size_t(c.groups) * c.rows);
            for (int g = 0; g < c.groups; ++g) {
                vg.row0[g] = g * c.rows; vg.nrows[g] = c.rows; vg.slot[g] = g;
                for (int r = 0; r < c.rows; ++r) lens[size_t(g) * c.rows + r] = L - c.rows + 1 + r;
            }
            q.memcpy(dlens, lens.data(), lens.size() * 4).wait();
            const int smax = verify_dpas_splits(base, vg, L);
            double us[4];
            auto timeit = [&](auto&& fn) {
                for (int i = 0; i < 3; ++i) fn();
                q.wait();
                const int N = 20;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) fn();
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            };
            us[0] = timeit([&] { bisect_attn<0>(q, base, vg, slot_bytes, dlens, smax, {}); });
            us[1] = timeit([&] { bisect_attn<1>(q, base, vg, slot_bytes, dlens, smax, {}); });
            us[2] = timeit([&] { bisect_attn<2>(q, base, vg, slot_bytes, dlens, smax, {}); });
            us[3] = timeit([&] { bisect_attn<3>(q, base, vg, slot_bytes, dlens, smax, {}); });
            double pf[3];
            pf[0] = timeit([&] { bisect_attn<4>(q, base, vg, slot_bytes, dlens, smax, {}); });
            pf[1] = timeit([&] { bisect_attn<5>(q, base, vg, slot_bytes, dlens, smax, {}); });
            pf[2] = timeit([&] { bisect_attn<6>(q, base, vg, slot_bytes, dlens, smax, {}); });
            const double kmaj = timeit([&] { bisect_attn<7>(q, base, vg, slot_bytes, dlens, smax, {}); });
            const double bytes = double(c.groups) * L * KVH * HD * 2;
            std::printf("%7d %2dx%-2d | %9.1f %9.1f %9.1f %9.1f | %8.1f | %d | prefetch 1/4/8: %9.1f %9.1f %9.1f | K key-major: %9.1f\n", L,
                        c.groups, c.rows, us[0], us[1], us[2], us[3], bytes / us[0] / 1e3, smax, pf[0], pf[1], pf[2], kmaj);
        }
    return 0;
}
