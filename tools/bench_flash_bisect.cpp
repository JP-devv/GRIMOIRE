// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_flash_bisect (2026-11-14) -- where does the prompt attention kernel (flash_esimd<256,128>) spend its time at
// long context?  One call = the last chunk of a 61K prompt: T query rows at positions start .. start+T-1, 24 heads,
// 4 KV heads, already-packed bf16 K / V / Q (the conversion kernels are separate and cheap).  Variants (MODE):
// 0 full | 1 no exp2 | 2 no QK DPAS | 3 no K loads | 4 no PV DPAS | 5 no V loads; work-groups of 8 / 16 / 32 threads.
//   bench_flash_bisect [T start]   (default 2270 59000)
// Build (grimoire-dev, repo root; 256 registers like libgrimoire_gemm.so):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -doubleGRF" \
//        -O3 -std=c++20 -I include -I src tools/bench_flash_bisect.cpp -o bin/bench_flash_bisect
#include "flash_bisect_kernels.inc"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace b70;

int main(int argc, char** argv) {
    constexpr int D = 256, BK = 128;
    const int H = 24, KVH = 4;
    const int T = argc > 1 ? std::atoi(argv[1]) : 2270;
    const int start = argc > 2 ? std::atoi(argv[2]) : 59000;
    const int kend = start + T, Sp = (kend + BK - 1) / BK * BK;
    const int Tp = (T + 63) / 64 * 64;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s  T=%d start=%d (keys up to %d)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), T, start, kend);
    const size_t kv = size_t(KVH) * Sp * D;
    std::vector<sycl_bf16> hk(kv), hq(size_t(H) * Tp * D);
    { std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 0.5f);
      for (auto& x : hk) x = sycl_bf16(nd(rng)); for (auto& x : hq) x = sycl_bf16(nd(rng) * 0.1f); }
    sycl_bf16* Kp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Vp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Qb = sycl::malloc_device<sycl_bf16>(hq.size(), q);
    float* out = sycl::malloc_device<float>(size_t(T) * H * D, q);
    q.memcpy(Kp, hk.data(), kv * 2); q.memcpy(Vp, hk.data(), kv * 2); q.memcpy(Qb, hq.data(), hq.size() * 2); q.wait();
    auto timeit = [&](auto&& fn) {
        fn(); q.wait();
        const int N = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) fn();
        q.wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
    };
    const double flop = 4.0 * H * D * double(T) * (start + T / 2.0);
    sycl::event e;
#define RUN(TPW, MODE) timeit([&] { flash_bisect<D, BK, TPW, MODE>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); })
    std::printf("%-5s | %8s %8s %8s %8s %8s %8s | TFLOP/s full\n", "TPW", "full", "noexp", "noQKdpas", "noKload", "noPVdpas", "noVload");
    {
        double m[6] = {RUN(8, 0), RUN(8, 1), RUN(8, 2), RUN(8, 3), RUN(8, 4), RUN(8, 5)};
        std::printf("%-5d | %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f | %.1f\n", 8, m[0], m[1], m[2], m[3], m[4], m[5], flop / m[0] / 1e9);
    }
    {
        double m0 = RUN(16, 0), m1 = RUN(32, 0);
        std::printf("full at TPW 16: %.2f ms, TPW 32: %.2f ms\n", m0, m1);
    }
    {
        double k6 = RUN(8, 6), k7 = RUN(8, 7);
        std::printf("K tile-major: %.2f ms (%.1f TFLOP/s), K+V tile-major: %.2f ms (%.1f TFLOP/s)\n", k6, flop / k6 / 1e9, k7, flop / k7 / 1e9);
        double k8 = RUN(8, 8), k9 = RUN(8, 9);
        {
            const size_t n = size_t(T) * H * D;
            std::vector<float> o0(n), o9(n);
            flash_bisect<D, BK, 8, 0>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); q.wait();
            q.memcpy(o0.data(), out, n * 4).wait();
            flash_bisect<D, BK, 8, 9>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); q.wait();
            q.memcpy(o9.data(), out, n * 4).wait();
            size_t ndiff = 0; for (size_t i = 0; i < n; ++i) ndiff += o0[i] != o9[i];
            std::printf("2 KB loads vs production: %zu of %zu outputs differ\n", ndiff, n);
        }
        std::printf("K loads 2 KB: %.2f ms (%.1f TFLOP/s), K+V loads 2 KB: %.2f ms (%.1f TFLOP/s)\n", k8, flop / k8 / 1e9, k9, flop / k9 / 1e9);
    }
    return 0;
}
