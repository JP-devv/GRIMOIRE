// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_flash_team (2026-11-15) -- prompt attention: production flash_esimd<256,128> (8 rows per thread, MODE 0 of
// tools/flash_bisect_kernels.inc) vs flash_team<256, BK, NT, TEAMS> (tools/flash_team_kernels.inc: NT threads share
// 8 * NT rows, K / V bytes per FLOP ~1/NT).  Qwen3.8-27B geometry: 24 heads, 4 KV heads, width 256, packed bf16
// K / V / Q as flash_fast_impl builds them.  Prints ms, TFLOP/s and max |team - production| / max |production|.
//   bench_flash_team            (shapes: tiny first, then T=2270 @59000, T=3840 @0, T=3840 @57600)
// Build (grimoire-dev, repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -doubleGRF" \
//        -O3 -std=c++20 -I include -I src tools/bench_flash_team.cpp -o bin/bench_flash_team
#include "flash_bisect_kernels.inc"
#include "flash_team_kernels.inc"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace b70;

int main() {
    constexpr int D = 256, BK = 128;
    const int H = 24, KVH = 4;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int shapes[][2] = {{100, 37}, {2270, 59000}, {3840, 0}, {3840, 57600}};
    const int maxk = 61440 + BK, maxT = 3840 + 64;
    const size_t kv = size_t(KVH) * maxk * D;
    sycl_bf16* Kp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Vp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Qb = sycl::malloc_device<sycl_bf16>(size_t(H) * maxT * D, q);
    float* out = sycl::malloc_device<float>(size_t(maxT) * H * D, q);
    std::vector<sycl_bf16> hk(kv), hv(kv), hq(size_t(H) * maxT * D);
    { std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.0f);
      for (auto& x : hk) x = sycl_bf16(nd(rng)); for (auto& x : hv) x = sycl_bf16(nd(rng));
      for (auto& x : hq) x = sycl_bf16(nd(rng) * 0.15f); }
    for (const auto& sh : shapes) {
        const int T = sh[0], start = sh[1], kend = start + T, Sp = (kend + BK - 1) / BK * BK;
        const int Tp = (T + 63) / 64 * 64;
        // the bench layouts depend on Sp / Tp: upload per shape (keys >= kend zeroed, like flash_fast_impl)
        std::vector<sycl_bf16> k2(size_t(KVH) * Sp * D), v2(size_t(KVH) * Sp * D), q2(size_t(H) * Tp * D);
        for (size_t i = 0; i < k2.size(); ++i) {
            const size_t s = (i / 2) % Sp;                       // K [KVH][D/2][Sp][2]
            k2[i] = s < size_t(kend) ? hk[i] : sycl_bf16(0.0f);
            const size_t sv = ((i / 2) / D % (Sp / 2)) * 2 + (i & 1);   // V [KVH][Sp/2][D][2]
            v2[i] = sv < size_t(kend) ? hv[i] : sycl_bf16(0.0f);
        }
        for (size_t i = 0; i < q2.size(); ++i) q2[i] = (i / D) % Tp < size_t(T) ? hq[i] : sycl_bf16(0.0f);
        q.memcpy(Kp, k2.data(), k2.size() * 2); q.memcpy(Vp, v2.data(), v2.size() * 2);
        q.memcpy(Qb, q2.data(), q2.size() * 2); q.wait();
        const size_t n = size_t(T) * H * D;
        std::vector<float> ref(n), got(n);
        auto timeit = [&](auto&& fn) {
            fn(); q.wait();
            const int N = T > 1000 ? 4 : 1;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) fn();
            q.wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
        };
        const double flop = 4.0 * H * D * double(T) * (start + T / 2.0);
        sycl::event e;
        q.memset(out, 0, n * 4).wait();
        const double t0 = timeit([&] { flash_bisect<D, BK, 8, 0>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); });
        q.memcpy(ref.data(), out, n * 4).wait();
        double mref = 0; for (float x : ref) mref = std::fmax(mref, std::fabs(x));
        std::printf("T=%d start=%d  production: %8.2f ms %6.1f TFLOP/s  (max|out| %.3f)\n", T, start, t0, flop / t0 / 1e9, mref);
        auto check = [&](const char* name, double ms) {
            q.memcpy(got.data(), out, n * 4).wait();
            double md = 0; size_t bad = 0;
            for (size_t i = 0; i < n; ++i) { const double d = std::fabs(double(got[i]) - ref[i]); md = std::fmax(md, d);
                bad += !(d <= 1e-3 * mref + 1e-6); }
            std::printf("  %-22s %8.2f ms %6.1f TFLOP/s  x%.2f  max|diff|/max|ref| %.2e  over 1e-3: %zu\n",
                        name, ms, flop / ms / 1e9, t0 / ms, md / (mref > 0 ? mref : 1), bad);
        };
#define TEAM(BKX, NT, TEAMS) do { q.memset(out, 0, n * 4).wait(); \
        const double ms = timeit([&] { flash_team<D, BKX, NT, TEAMS>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); }); \
        check("team BK" #BKX " NT" #NT " x" #TEAMS, ms); } while (0)
        TEAM(128, 4, 2);
        TEAM(128, 4, 4);
        TEAM(64, 4, 2);
        TEAM(64, 4, 4);
        // the 64-key block changes the softmax block max (so the bf16 P rounding): compare with production at 64
        {
            q.memset(out, 0, n * 4).wait();
            const double t64 = timeit([&] { flash_bisect<D, 64, 8, 0>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); });
            check("production BK64", t64);
            q.memcpy(ref.data(), out, n * 4).wait();
            TEAM(64, 4, 2);
        }
    }
    return 0;
}
