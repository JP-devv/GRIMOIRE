// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// onednn_u4_probe.cpp (2026-11-14) -- oneDNN matmul with INT4 weights decompression on Qwen3.8-27B shapes, against
// GRIMOIRE's own INT4 small-M kernel past 16 rows (int4_smallm_fdq_impl<4>: 32 rows at 230-340 GB/s, 16 rows at
// 430-550).  Weights in GRIMOIRE's INT4 layout: [N][K/2] nibbles (low nibble = even K), a bf16 scale and a u8 zero per
// 128 K, both [N][K/128].  oneDNN sees W as {K, N} u4 in tag ba; the scales / zeros as {K/128, N} -- tried in tag ba
// (GRIMOIRE's layout, zero copies) and tag ab (transposed copy), each checked against a host reference.
// Weights rotate over copies so they stream from DRAM.  M = 16 / 32 / 64 / 128 / 512 / 2048.
//
// Build (grimoire-dev):  icpx -fsycl -O3 -std=c++20 -I /onednn/include tools/onednn_u4_probe.cpp -L /onednn/lib -ldnnl \
//                          '-Wl,-rpath,$ORIGIN' -o bin/onednn_u4_probe
#include <sycl/sycl.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;
using bf16 = sycl::ext::oneapi::bfloat16;

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
    struct Shape { const char* nm; int N, K; };
    const std::vector<Shape> shapes = {{"gate_up", 34816, 5120}, {"down", 5120, 17408}, {"dn_qkv", 10240, 5120},
                                       {"z", 6144, 5120},        {"out/o", 5120, 6144}, {"attn_q", 12288, 5120},
                                       {"kv", 1024, 5120},       {"lm_head", 248320, 5120}};
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;
    const std::vector<int> Ms = {16, 32, 64, 128, 512, 2048};
    constexpr int G = 128, MMAX = 2048;
    std::mt19937 rng(7);
    for (int si = 0; si < int(shapes.size()); ++si) {
        if (only >= 0 && si != only) continue;
        const int N = shapes[si].N, K = shapes[si].K, KG = K / G;
        const size_t pb = size_t(N) * K / 2;
        const size_t wb = pb + size_t(N) * KG * 3;
        const int copies = std::max(2, std::min(8, int((1536ull << 20) / wb)));
        std::vector<uint8_t> hp(pb), hz(size_t(N) * KG), hzT(hz.size());
        std::vector<bf16> hs(size_t(N) * KG), hsT(hs.size());
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hz) v = uint8_t(rng() % 16);
        for (auto& v : hs) v = bf16(0.001f * float(1 + rng() % 16));
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < KG; ++g) { hzT[size_t(g) * N + n] = hz[size_t(n) * KG + g]; hsT[size_t(g) * N + n] = hs[size_t(n) * KG + g]; }
        std::vector<bf16> hx(size_t(MMAX) * K);
        for (auto& v : hx) v = bf16(float(int(rng() % 2001) - 1000) / 1000.0f);
        std::vector<uint8_t*> P(copies);
        for (auto& p : P) { p = sycl::malloc_device<uint8_t>(pb, q); q.memcpy(p, hp.data(), pb); }
        uint8_t* z = sycl::malloc_device<uint8_t>(hz.size(), q);
        uint8_t* zT = sycl::malloc_device<uint8_t>(hz.size(), q);
        bf16* s = sycl::malloc_device<bf16>(hs.size(), q);
        bf16* sT = sycl::malloc_device<bf16>(hs.size(), q);
        bf16* x = sycl::malloc_device<bf16>(hx.size(), q);
        float* y = sycl::malloc_device<float>(size_t(MMAX) * N, q);
        q.memcpy(z, hz.data(), hz.size()); q.memcpy(zT, hzT.data(), hzT.size());
        q.memcpy(s, hs.data(), hs.size() * 2); q.memcpy(sT, hsT.data(), hsT.size() * 2);
        q.memcpy(x, hx.data(), hx.size() * 2); q.wait();
        std::printf("%-8s N=%-6d K=%-5d  %.1f MB/copy x%d\n", shapes[si].nm, N, K, wb / 1e6, copies);
        // host reference for a few outputs
        auto ref = [&](int m, int n) {
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t b = hp[size_t(n) * (K / 2) + k / 2];
                const int qv = (k & 1) ? (b >> 4) : (b & 15);
                const int g = k / G;
                acc += double(float(hx[size_t(m) * K + k])) * double(qv - int(hz[size_t(n) * KG + g])) *
                       double(float(hs[size_t(n) * KG + g]));
            }
            return acc;
        };
        for (const bool grimoire_layout : {true, false}) {
            for (int M : Ms) {
                try {
                    dnnl::memory::desc src({M, K}, dt::bf16, tag::ab), wei({K, N}, dt::u4, tag::ba),
                        dst({M, N}, dt::f32, tag::ab);
                    dnnl::primitive_attr attr;
                    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {G, 1}, dt::bf16);
                    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {G, 1}, dt::u8);
                    attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
                    dnnl::matmul::primitive_desc pd(eng, src, wei, dst, attr);
                    const auto lt = grimoire_layout ? tag::ba : tag::ab;
                    dnnl::memory::desc smd({KG, N}, dt::bf16, lt), zmd({KG, N}, dt::u8, lt);
                    auto scr_md = pd.scratchpad_desc();
                    void* scr = scr_md.get_size() ? sycl::malloc_device(scr_md.get_size(), q) : nullptr;
                    dnnl::matmul prim(pd);
                    auto run = [&](int c) {
                        using namespace dnnl::sycl_interop;
                        std::unordered_map<int, dnnl::memory> a;
                        a[DNNL_ARG_SRC] = make_memory(src, eng, memory_kind::usm, x);
                        a[DNNL_ARG_WEIGHTS] = make_memory(wei, eng, memory_kind::usm, P[c]);
                        a[DNNL_ARG_DST] = make_memory(dst, eng, memory_kind::usm, y);
                        a[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
                            make_memory(smd, eng, memory_kind::usm, grimoire_layout ? (void*)s : (void*)sT);
                        a[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] =
                            make_memory(zmd, eng, memory_kind::usm, grimoire_layout ? (void*)z : (void*)zT);
                        if (scr) a[DNNL_ARG_SCRATCHPAD] = make_memory(scr_md, eng, memory_kind::usm, scr);
                        prim.execute(strm, a);
                    };
                    run(0); q.wait();
                    std::vector<float> hy(size_t(M) * N);
                    q.memcpy(hy.data(), y, hy.size() * 4).wait();
                    double md = 0, mx = 0;
                    for (int t = 0; t < 6; ++t) {
                        const int m = int(rng() % M), n = int(rng() % N);
                        const double r = ref(m, n);
                        md = std::max(md, std::fabs(double(hy[size_t(m) * N + n]) - r)); mx = std::max(mx, std::fabs(r));
                    }
                    for (int i = 0; i < 2 * copies; ++i) run(i % copies);
                    q.wait();
                    const int it = std::max(10, 3 * copies);
                    auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it; ++i) run(i % copies);
                    q.wait();
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
                    std::printf("  %s M=%-5d %9.1f us  %6.1f GB/s  %6.1f TFLOP/s  rel err %.1e  %s\n",
                                grimoire_layout ? "scales[N][K/G]" : "scales[K/G][N]", M, us, wb / us / 1e3,
                                2.0 * M * N * K / us / 1e6, mx > 0 ? md / mx : md, pd.impl_info_str());
                    if (scr) sycl::free(scr, q);
                } catch (const std::exception& e) {
                    std::printf("  %s M=%d: %s\n", grimoire_layout ? "scales[N][K/G]" : "scales[K/G][N]", M, e.what());
                }
            }
        }
        for (auto p : P) sycl::free(p, q);
        sycl::free(z, q); sycl::free(zT, q); sycl::free(s, q); sycl::free(sT, q); sycl::free(x, q); sycl::free(y, q);
    }
    return 0;
}
