// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// src/flash_team_i8.hpp (2026-11-17, experiment) -- flash_team (src/flash_team.hpp) with S = Q K^T on int8 XMX.
//
// Ian / Wrapzii 11-17: "use int8 for the KV cache, it is native for the XMX engine".  The cache is already 1 byte
// (FP8 E4M3); what is not 8-bit is the prompt attention's math: flash_fast_impl packs K / V to bf16 per call and the
// team kernel multiplies bf16 on XMX.  Measured 11-17 (v1.10.8, one B70, Swift INT4, 64K prompt, fenced regions):
// attention 12.6 of 34.6 s.  Here S = Q K^T runs as int8 x int8 (dpas<8, 8, int, int, int8, int8>: K-depth 32 a
// call, twice the bf16 rate): Q quantized per query row (its own scale), K per key (one scale a key and KV head),
// S = int32 sum * qs[row] * ks[key].  Softmax, P (bf16) and P V (bf16 XMX) are flash_team's, unchanged.
//
// Layouts (built by flash_i8_pack_q / flash_i8_pack_k):
//   Qi8 [H][Tp][D] int8, qs [H][Tp] f32         (Q already carries softmax scale * log2 e, as Qb does)
//   Ki8 [KVH][D/4][Sp] u32 (4 consecutive dims of one key per u32: the B operand's 4-deep int8 VNNI), ks [KVH][Sp]
#pragma once
#include "kernels.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
#include <stdexcept>
namespace b70 {

// Q bf16 [H][Tp][D] -> int8 + one f32 scale a row (max |q| / 127; a zero row gets scale 1).
inline sycl::event flash_i8_pack_q(sycl::queue& q, const sycl_bf16* Qb, int8_t* Qi8, float* qs, int H, int Tp, int D,
                                   const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(H) * Tp * 16, 16), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const size_t row = it.get_group(0);
            const int lane = int(sg.get_local_id()[0]);
            const sycl_bf16* x = Qb + row * D;
            float m = 0.f;
            for (int d = lane; d < D; d += 16) m = sycl::fmax(m, sycl::fabs(float(x[d])));
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
            if (lane == 0) qs[row] = s;
            for (int d = lane; d < D; d += 16)
                Qi8[row * D + d] = int8_t(sycl::clamp(sycl::rint(float(x[d]) * inv), -127.f, 127.f));
        });
    });
}

// K bf16 [KVH][D/2][Sp][2] (flash_fast_impl's pack) -> int8 [KVH][D/4][Sp] u32 + one f32 scale a key.
inline sycl::event flash_i8_pack_k(sycl::queue& q, const sycl_bf16* Kp, uint32_t* Ki8, float* ks, int KVH, int Sp, int D,
                                   const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(KVH), size_t(Sp)), [=](sycl::id<2> id) {
            const int kv = int(id[0]), s = int(id[1]);
            const sycl_bf16* base = Kp + size_t(kv) * (D / 2) * Sp * 2;
            auto at = [&](int d) { return float(base[(size_t(d / 2) * Sp + s) * 2 + (d & 1)]); };
            float m = 0.f;
            for (int d = 0; d < D; ++d) m = sycl::fmax(m, sycl::fabs(at(d)));
            const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
            ks[size_t(kv) * Sp + s] = sc;
            uint32_t* o = Ki8 + size_t(kv) * (D / 4) * Sp;
            for (int d4 = 0; d4 < D / 4; ++d4) {
                uint32_t w = 0;
                for (int j = 0; j < 4; ++j) {
                    const int v = int(sycl::clamp(sycl::rint(at(d4 * 4 + j) * inv), -127.f, 127.f));
                    w |= (uint32_t(v) & 0xFFu) << (8 * j);
                }
                o[size_t(d4) * Sp + s] = w;
            }
        });
    });
}

// flash_team with int8 S.  Same template parameters and output as flash_team<D, BK, NT, TEAMS> (no GQA variant).
template <int D, int BK, int NT, int TEAMS, int RTX = 8 * NT>
sycl::event flash_team_i8(sycl::queue& q, sycl::event dep, const int8_t* Qi8, const float* qs, int Tp,
                          const uint32_t* Ki8, const float* ks, const sycl_bf16* Vp, int Sp, float* out,
                          int tokens, int start, int H, int KVH) {
    namespace es = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    constexpr int RT = RTX, NRB = RT / 8, DP = D / NT, DPT = DP / 16, KS = BK / NT, KC = KS / 16;
    constexpr int NBK = BK / 16, DT8 = D / 32, WGT = NT * TEAMS, WROWS = RT * TEAMS;
    static_assert(KS % 16 == 0 && DP % 16 == 0 && RT <= 32 && RT % 8 == 0 && D % 32 == 0, "team geometry");
    constexpr int MAXB = NT * RT * 4;
    constexpr int PB = BK * RT * 2;
    constexpr int TEAM_SLM = MAXB + PB;
    const int qtiles = (tokens + WROWS - 1) / WROWS;
    const size_t nwg = size_t(qtiles) * H;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<1>(nwg * WGT, WGT), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<TEAM_SLM * TEAMS>();
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            const int g = int(it.get_group(0)), lt = int(it.get_local_id(0));
            const int team = lt / NT, t = lt % NT;
            const int hq = g % H;
            const int qt = qtiles - 1 - g / H, kvh = hq / (H / KVH);
            const int wr0 = qt * WROWS, r0 = wr0 + team * RT;
            const int wlast = tokens - 1 < wr0 + WROWS - 1 ? tokens - 1 : wr0 + WROWS - 1;
            const int kmax = start + wlast + 1;
            const uint32_t smax = uint32_t(team * TEAM_SLM), sp = smax + MAXB;
            const int8_t* qh = Qi8 + size_t(hq) * Tp * D;
            const uint32_t* kh = Ki8 + size_t(kvh) * (D / 4) * Sp;
            const float* ksh = ks + size_t(kvh) * Sp;
            const uint32_t* vh = reinterpret_cast<const uint32_t*>(Vp + size_t(kvh) * (Sp / 2) * D * 2);
            // this team's row scales (rows past Tp read padding rows: Tp is a whole number of tiles)
            es::simd<float, RT> qsr = es::block_load<float, RT>(qs + size_t(hq) * Tp + r0);
            es::simd<float, RT * DP> o = 0.0f;
            es::simd<float, RT> m = NINF, l = 0.0f;
            const es::simd<int, 16> col(0, 1);
            for (int s0 = 0; s0 < kmax; s0 += BK) {
                const int ks0 = s0 + t * KS;
                es::simd<int, KS * RT> si = 0;          // [c][rb][8][16] int32
                #pragma unroll 2
                for (int dt = 0; dt < DT8; ++dt) {
                    es::simd<int8_t, RT * 32> a = es::load_2d<int8_t, 32, RT>(
                        qh, D - 1, Tp - 1, D - 1, dt * 32, r0);
                    #pragma unroll
                    for (int c = 0; c < KC; ++c) {
                        es::simd<uint32_t, 128> keys = es::load_2d<uint32_t, 16, 8>(
                            kh, Sp * 4 - 1, D / 4 - 1, Sp * 4 - 1, ks0 + c * 16, dt * 8);
                        #pragma unroll
                        for (int rb = 0; rb < NRB; ++rb)
                            si.template select<128, 1>((c * NRB + rb) * 128) =
                                xmx::dpas<8, 8, int, int, int8_t, int8_t>(
                                    es::simd<int, 128>(si.template select<128, 1>((c * NRB + rb) * 128)),
                                    keys.template bit_cast_view<int8_t>().read(),
                                    es::simd<int8_t, 256>(a.template select<256, 1>(rb * 256)));
                    }
                }
                es::simd<float, KS * RT> sc;
                #pragma unroll
                for (int c = 0; c < KC; ++c) {
                    const es::simd<float, 16> kss = es::block_load<float, 16>(ksh + ks0 + c * 16);
                    #pragma unroll
                    for (int rb = 0; rb < NRB; ++rb)
                        #pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int idx = ((c * NRB + rb) * 8 + r) * 16;
                            sc.template select<16, 1>(idx) =
                                es::simd<float, 16>(si.template select<16, 1>(idx)) * kss * float(qsr[rb * 8 + r]);
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
