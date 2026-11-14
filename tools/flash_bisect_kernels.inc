// tools/flash_bisect_kernels.inc (2026-11-14): src/gemm_fast.cpp's flash_esimd with parts switched off --
// MODE 0 full | 1 no exp2 | 2 no QK DPAS | 3 no K loads | 4 no PV DPAS | 5 no V loads | 6 K tile-major | 7 K+V tile-major.
// Timing only for 1..7 (the bench packs nothing in the tile layouts).
#include "kernels.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
namespace b70 {
template <int D, int BK, int TPW, int MODE>
sycl::event flash_bisect(sycl::queue& q, sycl::event dep, const sycl_bf16* Qb, int Tp,
                        const sycl_bf16* Kp, const sycl_bf16* Vp, int Sp, float* out,
                        int tokens, int start, int H, int KVH) {
    namespace es = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    constexpr int RB = 8, DT = D / 16, NB = BK / 16;
    const int qtiles = (tokens + RB * TPW - 1) / (RB * TPW);
    const size_t nthreads = size_t(qtiles) * H * TPW;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<1>(nthreads, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            const int g = int(it.get_group(0)), lt = int(it.get_local_id(0));
            const int hq = g % H, qt = qtiles - 1 - g / H, kvh = hq / (H / KVH);
            const int r0 = (qt * TPW + lt) * RB;
            if (r0 >= tokens) return;
            const int lastr = tokens - 1 < r0 + RB - 1 ? tokens - 1 : r0 + RB - 1;
            const int kmax = start + lastr + 1;
            const sycl_bf16* qh = Qb + size_t(hq) * Tp * D;
            const uint32_t* kh = reinterpret_cast<const uint32_t*>(Kp + size_t(kvh) * (D / 2) * Sp * 2);
            const uint32_t* vh = reinterpret_cast<const uint32_t*>(Vp + size_t(kvh) * (Sp / 2) * D * 2);
            es::simd<float, RB * D> o = 0.0f;
            es::simd<float, RB> m = NINF, l = 0.0f;
            const es::simd<int, 16> col(0, 1);
            for (int s0 = 0; s0 < kmax; s0 += BK) {
                // Key tiles stay block-major: each 16-key chunk is an 8x16
                // DPAS accumulator, so every register offset is constant.
                es::simd<float, NB * RB * 16> scores = 0.0f;
                #pragma unroll 2
                for (int dt = 0; dt < DT; ++dt) {
                    es::simd<sycl_bf16, RB * 16> a = es::load_2d<sycl_bf16, 16, RB>(
                        qh, D * 2 - 1, Tp - 1, D * 2 - 1, dt * 16, r0);
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<uint32_t, 128> keys;
                        if constexpr (MODE == 3) keys = 0x3c003c00u;           // no K loads
                        else if constexpr (MODE == 6 || MODE == 7)               // K tile-major: [Sp/16][D/2][16 keys] u32
                            keys = es::load_2d<uint32_t, 16, 8>(
                                kh + size_t((s0 + b * 16) / 16) * (D / 2) * 16, 63, D / 2 - 1, 63, 0, dt * 8);
                        else keys = es::load_2d<uint32_t, 16, 8>(
                            kh, Sp * 4 - 1, D / 2 - 1, Sp * 4 - 1, s0 + b * 16, dt * 8);
                        if constexpr (MODE == 2) {                               // no QK matrix work
                            es::simd<float, RB * 16> kf = keys.template bit_cast_view<sycl_bf16>().read().template select<RB * 16, 1>(0);
                            scores.template select<RB * 16, 1>(b * RB * 16) += kf;
                        } else
                        scores.template select<RB * 16, 1>(b * RB * 16) =
                            xmx::dpas<8, RB, float, float, sycl_bf16, sycl_bf16>(
                                es::simd<float, RB * 16>(scores.template select<RB * 16, 1>(b * RB * 16)),
                                keys.template bit_cast_view<sycl_bf16>().read(), a);
                    }
                }
                if (s0 + BK - 1 > start + r0) {
                    #pragma unroll
                    for (int b = 0; b < NB; ++b)
                        #pragma unroll
                        for (int r = 0; r < RB; ++r) {
                            const int lim = r0 + r < tokens ? start + r0 + r : -1;
                            es::simd<float, 16> v = scores.template select<16, 1>((b * RB + r) * 16);
                            v.merge(es::simd<float, 16>(NINF), col + s0 + b * 16 > lim);
                            scores.template select<16, 1>((b * RB + r) * 16) = v;
                        }
                }
                es::simd<float, RB> bm;
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    es::simd<float, 16> v = scores.template select<16, 1>(r * 16);
                    #pragma unroll
                    for (int b = 1; b < NB; ++b)
                        v = es::max(v, es::simd<float, 16>(scores.template select<16, 1>((b * RB + r) * 16)));
                    bm[r] = es::hmax<float>(v);
                }
                es::simd<float, RB> mn = es::max(m, bm), ms = mn;
                ms.merge(es::simd<float, RB>(0.0f), mn == NINF);
                es::simd<float, RB> corr = es::exp2(m - ms), rs;
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const float msr = ms[r];
                    es::simd<float, 16> sum = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<float, 16> prob;
                        if constexpr (MODE == 1) prob = es::simd<float, 16>(scores.template select<16, 1>((b * RB + r) * 16)) - msr;
                        else prob = es::exp2(
                            es::simd<float, 16>(scores.template select<16, 1>((b * RB + r) * 16)) - msr);
                        scores.template select<16, 1>((b * RB + r) * 16) = prob;
                        sum += prob;
                    }
                    rs[r] = es::reduce<float>(sum, std::plus<>());
                }
                l = l * corr + rs;
                m = mn;
                float cr[RB];
                #pragma unroll
                for (int r = 0; r < RB; ++r) cr[r] = corr[r];
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt)
                    #pragma unroll
                    for (int r = 0; r < RB; ++r)
                        o.template select<16, 1>((dt * RB + r) * 16) *= cr[r];
                es::simd<sycl_bf16, NB * RB * 16> probabilities = scores;
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt) {
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<uint32_t, 128> values;
                        if constexpr (MODE == 5) values = 0x3c003c00u;         // no V loads
                        else if constexpr (MODE == 7)                            // V tile-major: [Sp/16][D/16][8 pairs][16 dims] u32
                            values = es::load_2d<uint32_t, 16, 8>(
                                vh + size_t((s0 + b * 16) / 16) * (D / 16) * 128, 63, D - 1, 63, 0, dt * 8);
                        else values = es::load_2d<uint32_t, 16, 8>(
                            vh, D * 4 - 1, Sp / 2 - 1, D * 4 - 1, dt * 16, (s0 + b * 16) / 2);
                        if constexpr (MODE == 4) {                               // no PV matrix work
                            es::simd<float, RB * 16> vf = values.template bit_cast_view<sycl_bf16>().read().template select<RB * 16, 1>(0);
                            o.template select<RB * 16, 1>(dt * RB * 16) += vf;
                        } else
                        o.template select<RB * 16, 1>(dt * RB * 16) =
                            xmx::dpas<8, RB, float, float, sycl_bf16, sycl_bf16>(
                                es::simd<float, RB * 16>(o.template select<RB * 16, 1>(dt * RB * 16)),
                                values.template bit_cast_view<sycl_bf16>().read(),
                                es::simd<sycl_bf16, RB * 16>(probabilities.template select<RB * 16, 1>(b * RB * 16)));
                    }
                }
            }
            es::simd<float, RB> inv = 1.0f / l;
            inv.merge(es::simd<float, RB>(0.0f), l == 0.0f);
            float iv[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) iv[r] = inv[r];
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                #pragma unroll
                for (int r = 0; r < RB; ++r)
                    o.template select<16, 1>((dt * RB + r) * 16) *= iv[r];
                es::store_2d<float, 16, RB>(out, H * D * 4 - 1, tokens - 1, H * D * 4 - 1,
                    hq * D + dt * 16, r0, es::simd<float, RB * 16>(o.template select<RB * 16, 1>(dt * RB * 16)));
            }
        });
    });
}

} // namespace b70
