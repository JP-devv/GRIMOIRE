// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_flash_i8 (2026-11-17) -- prompt attention, S = Q K^T on int8 XMX (src/flash_team_i8.hpp) against the
// production bf16 team kernel flash_team<256, 64, 4, 2> (src/flash_team.hpp), Qwen3.8-27B geometry (24 heads,
// 4 KV heads, width 256).  Two data sets: plain normal values, and K with outlier channels (8 of 256 dims x12) --
// what per-key int8 scales have to survive.  Prints ms, TFLOP/s, the int8 packs' ms, max |diff| / max |ref| and the
// RMS relative error of the attention output.
// Build (grimoire-dev, repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -doubleGRF" \
//        -O3 -std=c++20 -I include -I src tools/bench_flash_i8.cpp -o bin/bench_flash_i8
#include "flash_team.hpp"
#include "flash_team_i8.hpp"
#include "b70/formats.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace b70;

int main() {
    constexpr int D = 256, BK = 64;
    const int H = 24, KVH = 4;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    // tiny first (a wrong kernel fails fast), then a 4K prompt chunk, the last chunk of a 64K prompt, of a 128K prompt
    const int shapes[][2] = {{100, 37}, {3840, 0}, {8192, 57344}, {8192, 122880}};
    const int maxk = 131072 + 256, maxT = 8192 + 64;
    const size_t kv = size_t(KVH) * maxk * D;
    sycl_bf16* Kp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Vp = sycl::malloc_device<sycl_bf16>(kv, q);
    sycl_bf16* Qb = sycl::malloc_device<sycl_bf16>(size_t(H) * maxT * D, q);
    int8_t* Qi8 = sycl::malloc_device<int8_t>(size_t(H) * maxT * D, q);
    float* qs = sycl::malloc_device<float>(size_t(H) * maxT, q);
    uint32_t* Ki8 = sycl::malloc_device<uint32_t>(kv / 4, q);
    float* ks = sycl::malloc_device<float>(size_t(KVH) * maxk, q);
    float* out = sycl::malloc_device<float>(size_t(maxT) * H * D, q);
    if (!Kp || !Vp || !Qb || !Qi8 || !qs || !Ki8 || !ks || !out) { std::printf("allocation failed\n"); return 1; }
    for (int outliers = 0; outliers < 2; ++outliers) {
        std::printf("== data: %s\n", outliers ? "K with 8 outlier channels (x12)" : "plain normal");
        for (const auto& sh : shapes) {
            const int T = sh[0], start = sh[1], kend = start + T, Sp = (kend + 255) / 256 * 256;
            const int Tp = (T + 63) / 64 * 64;
            std::mt19937 rng(5 + outliers);
            std::normal_distribution<float> nd(0.f, 1.0f);
            // exact (bf16) K / V, and the same values through the FP8 E4M3 cache as the engine stores them today
            std::vector<sycl_bf16> k2(size_t(KVH) * Sp * D), v2(size_t(KVH) * Sp * D), q2(size_t(H) * Tp * D);
            std::vector<sycl_bf16> k8(k2.size()), v8(v2.size());
            auto f8 = [](float x) { return sycl_bf16(e4m3_to_f32(f32_to_e4m3(x))); };
            for (size_t i = 0; i < k2.size(); ++i) {
                const size_t s = (i / 2) % Sp;                              // K [KVH][D/2][Sp][2]
                const int d = int(((i / 2) / Sp) % (D / 2)) * 2 + int(i & 1);
                const float boost = outliers && (d % 32 == 7) ? 12.f : 1.f;
                const float kx = s < size_t(kend) ? nd(rng) * boost : 0.0f;
                k2[i] = sycl_bf16(kx); k8[i] = f8(kx);
                const size_t sv = ((i / 2) / D % (Sp / 2)) * 2 + (i & 1);   // V [KVH][Sp/2][D][2]
                const float vx = sv < size_t(kend) ? nd(rng) : 0.0f;
                v2[i] = sycl_bf16(vx); v8[i] = f8(vx);
            }
            for (size_t i = 0; i < q2.size(); ++i)
                q2[i] = (i / D) % Tp < size_t(T) ? sycl_bf16(nd(rng) * 0.15f) : sycl_bf16(0.0f);
            q.memcpy(Kp, k2.data(), k2.size() * 2); q.memcpy(Vp, v2.data(), v2.size() * 2);
            q.memcpy(Qb, q2.data(), q2.size() * 2); q.wait();
            const size_t n = size_t(T) * H * D;
            std::vector<float> ref(n), got(n);
            auto timeit = [&](auto&& fn) {
                fn(); q.wait();
                const int N = T > 1000 ? 3 : 1;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) fn();
                q.wait();
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
            };
            const double flop = 4.0 * H * D * double(T) * (start + T / 2.0);
            sycl::event e;
            q.memset(out, 0, n * 4).wait();
            const double tb = timeit([&] { flash_team<D, BK, 4, 2>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); });
            q.memcpy(ref.data(), out, n * 4).wait();                 // reference: exact K / V
            double mref = 0; for (float x : ref) mref = std::fmax(mref, std::fabs(x));
            auto err = [&](const std::vector<float>& g, double& mx, double& rms) {
                double md = 0, se = 0, sr = 0;
                for (size_t i = 0; i < n; ++i) { const double d = double(g[i]) - ref[i];
                    md = std::fmax(md, std::fabs(d)); se += d * d; sr += double(ref[i]) * ref[i]; }
                mx = md / (mref > 0 ? mref : 1); rms = sr > 0 ? std::sqrt(se / sr) : 0.0;
            };
            // TODAY: the FP8 cache (K and V through E4M3), bf16 kernel
            q.memcpy(Kp, k8.data(), k8.size() * 2); q.memcpy(Vp, v8.data(), v8.size() * 2); q.wait();
            q.memset(out, 0, n * 4).wait();
            flash_team<D, BK, 4, 2>(q, e, Qb, Tp, Kp, Vp, Sp, out, T, start, H, KVH); q.wait();
            std::vector<float> today(n); q.memcpy(today.data(), out, n * 4).wait();
            double tmx, trms; err(today, tmx, trms);
            // int8 S: K int8 per key from the EXACT keys (as a cache write would), V through FP8 as today
            q.memcpy(Kp, k2.data(), k2.size() * 2).wait();
            const double tp = timeit([&] {
                flash_i8_pack_q(q, Qb, Qi8, qs, H, Tp, D);
                flash_i8_pack_k(q, Kp, Ki8, ks, KVH, Sp, D);
            });
            q.memset(out, 0, n * 4).wait();
            const double ti = timeit([&] {
                flash_team_i8<D, BK, 4, 2>(q, e, Qi8, qs, Tp, Ki8, ks, Vp, Sp, out, T, start, H, KVH);
            });
            q.memcpy(got.data(), out, n * 4).wait();
            double imx, irms; err(got, imx, irms);
            std::printf("T=%5d start=%6d  bf16 team %8.2f ms | int8-S %8.2f ms x%.2f (+ packs %.2f ms) | error vs exact K/V, "
                        "max / rms: TODAY (FP8 cache) %.2e / %.2e   INT8 keys %.2e / %.2e\n",
                        T, start, tb, ti, tb / ti, tp, tmx, trms, imx, irms);
            q.memcpy(Vp, v2.data(), v2.size() * 2).wait();           // exact V again for the next shape's reference
        }
    }
    return 0;
}
