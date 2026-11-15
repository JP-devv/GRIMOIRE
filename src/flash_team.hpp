// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// src/flash_team.hpp -- prompt attention by thread teams (included by gemm_fast.cpp and tools/bench_flash_team.cpp).
#pragma once
// (2026-11-15) prompt attention with TEAMS of NT threads sharing RT = 8 * NT query rows.
//
// Why: flash_esimd (src/gemm_fast.cpp) gives each thread 8 query rows of a head; every 16-key K / V tile it loads
// (512 B) feeds 8 rows only, ~8 FLOP per loaded byte.  At 61K that kernel ran at ~33 TFLOP/s and took 22.8 s of a
// 51 s prompt (unitrace 11-14); with no K loads it ran 2.5x faster (bench_flash_bisect MODE 3): load-bound.
// A team of NT threads owns RT = 8 * NT rows.  Per BK-key block, thread t computes S for all RT rows over its own
// BK / NT keys (all D dims), the team exchanges the per-row block max through SLM (barrier), each thread writes its
// P slice (bf16) to SLM (barrier), then thread t accumulates O[RT rows][its D / NT dims] over all BK keys.
// O stays RT x D / NT fp32 = 2048 floats (128 GRF) whatever NT is; the bytes loaded per FLOP fall ~NT-fold for K / V.
// Math = flash_esimd's (same S, same block max, same exp2, same bf16 P, same PV order per dim); only the softmax
// denominator l is summed per key slice and added at the end (fp32 reassociation).
#include "kernels.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
namespace b70 {
// RTX (2026-11-15): rows per team, default 8 * NT.  RTX < 8 * NT gives each thread fewer output dims (DP = D / NT)
// and more keys of S per Q load (KS = BK / NT): e.g. <D, 256, 8, 2, 32> = 32 rows, 8 threads, 32 keys each.
template <int D, int BK, int NT, int TEAMS, int RTX = 8 * NT>
sycl::event flash_team(sycl::queue& q, sycl::event dep, const sycl_bf16* Qb, int Tp,
                       const sycl_bf16* Kp, const sycl_bf16* Vp, int Sp, float* out,
                       int tokens, int start, int H, int KVH) {
    namespace es = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    constexpr int RT = RTX, NRB = RT / 8, DP = D / NT, DPT = DP / 16, KS = BK / NT, KC = KS / 16;
    constexpr int NBK = BK / 16, DT = D / 16, WGT = NT * TEAMS, WROWS = RT * TEAMS, QU = KC >= 2 ? 1 : 2;
    static_assert(KS % 16 == 0 && DP % 16 == 0 && RT <= 32 && RT % 8 == 0, "team geometry");
    constexpr int MAXB = NT * RT * 4;                 // per-row block max (later l) of every thread
    constexpr int PB = BK * RT * 2;                   // P [kc][rb][8][16] bf16
    constexpr int TEAM_SLM = MAXB + PB;
    const int qtiles = (tokens + WROWS - 1) / WROWS;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<1>(size_t(qtiles) * H * WGT, WGT), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<TEAM_SLM * TEAMS>();
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            const int g = int(it.get_group(0)), lt = int(it.get_local_id(0));
            const int team = lt / NT, t = lt % NT;
            const int hq = g % H, qt = qtiles - 1 - g / H, kvh = hq / (H / KVH);
            const int wr0 = qt * WROWS, r0 = wr0 + team * RT;
            const int wlast = tokens - 1 < wr0 + WROWS - 1 ? tokens - 1 : wr0 + WROWS - 1;
            const int kmax = start + wlast + 1;       // same for the whole work-group: barriers inside the loop
            const uint32_t smax = uint32_t(team * TEAM_SLM), sp = smax + MAXB;
            const sycl_bf16* qh = Qb + size_t(hq) * Tp * D;
            const uint32_t* kh = reinterpret_cast<const uint32_t*>(Kp + size_t(kvh) * (D / 2) * Sp * 2);
            const uint32_t* vh = reinterpret_cast<const uint32_t*>(Vp + size_t(kvh) * (Sp / 2) * D * 2);
            es::simd<float, RT * DP> o = 0.0f;        // [rb][dti][8][16]
            es::simd<float, RT> m = NINF, l = 0.0f;
            const es::simd<int, 16> col(0, 1);
            for (int s0 = 0; s0 < kmax; s0 += BK) {
                const int ks0 = s0 + t * KS;
                es::simd<float, KS * RT> sc = 0.0f;   // [c][rb][8][16]
                // BK 128 (KC 2) spills 1,280 bytes with this at 2 or 1 (11-15 build log), so the server uses BK 64
                #pragma unroll QU
                for (int dt = 0; dt < DT; ++dt) {
                    es::simd<sycl_bf16, RT * 16> a = es::load_2d<sycl_bf16, 16, RT>(
                        qh, D * 2 - 1, Tp - 1, D * 2 - 1, dt * 16, r0);
                    #pragma unroll
                    for (int c = 0; c < KC; ++c) {
                        es::simd<uint32_t, 128> keys = es::load_2d<uint32_t, 16, 8>(
                            kh, Sp * 4 - 1, D / 2 - 1, Sp * 4 - 1, ks0 + c * 16, dt * 8);
                        #pragma unroll
                        for (int rb = 0; rb < NRB; ++rb)
                            sc.template select<128, 1>((c * NRB + rb) * 128) =
                                xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    es::simd<float, 128>(sc.template select<128, 1>((c * NRB + rb) * 128)),
                                    keys.template bit_cast_view<sycl_bf16>().read(),
                                    es::simd<sycl_bf16, 128>(a.template select<128, 1>(rb * 128)));
                    }
                }
                if (ks0 + KS - 1 > start + r0) {
                    #pragma unroll
                    for (int c = 0; c < KC; ++c)
                        #pragma unroll
                        for (int rb = 0; rb < NRB; ++rb)
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const int row = r0 + rb * 8 + r;
                                const int lim = row < tokens ? start + row : -1;
                                es::simd<float, 16> v = sc.template select<16, 1>(((c * NRB + rb) * 8 + r) * 16);
                                v.merge(es::simd<float, 16>(NINF), col + ks0 + c * 16 > lim);
                                sc.template select<16, 1>(((c * NRB + rb) * 8 + r) * 16) = v;
                            }
                }
                es::simd<float, RT> bm;
                #pragma unroll
                for (int rb = 0; rb < NRB; ++rb)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        es::simd<float, 16> v = sc.template select<16, 1>((rb * 8 + r) * 16);
                        #pragma unroll
                        for (int c = 1; c < KC; ++c)
                            v = es::max(v, es::simd<float, 16>(sc.template select<16, 1>(((c * NRB + rb) * 8 + r) * 16)));
                        bm[rb * 8 + r] = es::hmax<float>(v);
                    }
                es::slm_block_store<float, RT>(smax + t * RT * 4, bm);
                es::barrier();
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    bm = es::max(bm, es::slm_block_load<float, RT>(smax + j * RT * 4));
                es::simd<float, RT> mn = es::max(m, bm), ms = mn;
                ms.merge(es::simd<float, RT>(0.0f), mn == NINF);
                es::simd<float, RT> corr = es::exp2(m - ms), rs;
                #pragma unroll
                for (int rb = 0; rb < NRB; ++rb)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float msr = ms[rb * 8 + r];
                        es::simd<float, 16> sum = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < KC; ++c) {
                            es::simd<float, 16> prob = es::exp2(
                                es::simd<float, 16>(sc.template select<16, 1>(((c * NRB + rb) * 8 + r) * 16)) - msr);
                            sc.template select<16, 1>(((c * NRB + rb) * 8 + r) * 16) = prob;
                            sum += prob;
                        }
                        rs[rb * 8 + r] = es::reduce<float>(sum, std::plus<>());
                    }
                l = l * corr + rs;
                m = mn;
                // one 8 x 16 tile at a time: a whole bf16 copy of sc (+ its u32 view) next to o spilled at BK 128
                #pragma unroll
                for (int i = 0; i < KS * RT / 128; ++i) {
                    es::simd<sycl_bf16, 128> pt = sc.template select<128, 1>(i * 128);
                    es::slm_block_store<uint32_t, 64>(sp + (t * KS * RT + i * 128) * 2,
                                                      pt.template bit_cast_view<uint32_t>().read());
                }
                float cr[RT];
                #pragma unroll
                for (int r = 0; r < RT; ++r) cr[r] = corr[r];
                #pragma unroll
                for (int rb = 0; rb < NRB; ++rb)
                    #pragma unroll
                    for (int dti = 0; dti < DPT; ++dti)
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            o.template select<16, 1>(((rb * DPT + dti) * 8 + r) * 16) *= cr[rb * 8 + r];
                es::barrier();
                for (int kc = 0; kc < NBK; ++kc) {
                    es::simd<uint32_t, NRB * 64> pa;
                    #pragma unroll
                    for (int rb = 0; rb < NRB; ++rb)
                        pa.template select<64, 1>(rb * 64) = es::slm_block_load<uint32_t, 64>(sp + (kc * NRB + rb) * 256);
                    #pragma unroll
                    for (int dti = 0; dti < DPT; ++dti) {
                        es::simd<uint32_t, 128> values = es::load_2d<uint32_t, 16, 8>(
                            vh, D * 4 - 1, Sp / 2 - 1, D * 4 - 1, t * DP + dti * 16, (s0 + kc * 16) / 2);
                        #pragma unroll
                        for (int rb = 0; rb < NRB; ++rb) {
                            es::simd<uint32_t, 64> pt = pa.template select<64, 1>(rb * 64);
                            o.template select<128, 1>((rb * DPT + dti) * 128) =
                                xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    es::simd<float, 128>(o.template select<128, 1>((rb * DPT + dti) * 128)),
                                    values.template bit_cast_view<sycl_bf16>().read(),
                                    pt.template bit_cast_view<sycl_bf16>().read());
                        }
                    }
                }
            }
            // softmax denominators: every thread holds l over its own key slices -- sum them in a fixed order
            es::slm_block_store<float, RT>(smax + t * RT * 4, l);
            es::barrier();
            es::simd<float, RT> lt_ = es::slm_block_load<float, RT>(smax);
            #pragma unroll
            for (int j = 1; j < NT; ++j) lt_ += es::slm_block_load<float, RT>(smax + j * RT * 4);
            if (r0 >= tokens) return;
            es::simd<float, RT> inv = 1.0f / lt_;
            inv.merge(es::simd<float, RT>(0.0f), lt_ == 0.0f);
            float iv[RT];
            #pragma unroll
            for (int r = 0; r < RT; ++r) iv[r] = inv[r];
            #pragma unroll
            for (int rb = 0; rb < NRB; ++rb)
                #pragma unroll
                for (int dti = 0; dti < DPT; ++dti) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r)
                        o.template select<16, 1>(((rb * DPT + dti) * 8 + r) * 16) *= iv[rb * 8 + r];
                    es::store_2d<float, 16, 8>(out, H * D * 4 - 1, tokens - 1, H * D * 4 - 1,
                        hq * D + t * DP + dti * 16, r0 + rb * 8,
                        es::simd<float, 128>(o.template select<128, 1>((rb * DPT + dti) * 128)));
                }
        });
    });
}
} // namespace b70
