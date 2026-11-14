// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// bench_kvread (2026-11-14) -- the decode attention's MEMORY pattern alone, no math: each thread walks its split of
// one KV head's keys 16 at a time and reads that block's K bytes (256 dims x 16 keys = 4 KB) and V bytes (4 KB),
// XOR-folding them into one register so nothing is optimised away.  K is read either as GRIMOIRE stores it today --
// [kv_head][256 dims][cap] (16 bytes from each of 256 rows, rows cap bytes apart) -- or key-major [kv_head][cap][256]
// (one contiguous 4 KB).  V is [kv_head][cap][256] in both.  Same grid as the verify kernel's 1x1 and 4x1 cases.
//   bench_kvread [L threads...]
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace es = sycl::ext::intel::esimd;

template <bool KEYMAJOR>
double run(sycl::queue& q, const uint8_t* k, const uint8_t* v, uint32_t* sink, int users, int L, int cap, int splits) {
    constexpr int KVH = 4, HD = 256;
    const size_t n = size_t(users) * KVH * splits;
    const int per = ((L + splits - 1) / splits + 15) / 16 * 16;
    auto go = [&] {
        q.parallel_for(sycl::nd_range<1>(n, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            int id = int(it.get_global_id(0));
            const int part = id % splits; id /= splits;
            const int kvh = id % KVH, u = id / KVH;
            const size_t slot = size_t(u) * KVH * HD * cap;
            const uint32_t* kh = reinterpret_cast<const uint32_t*>(k + slot + size_t(kvh) * HD * cap);
            const uint32_t* vh = reinterpret_cast<const uint32_t*>(v + slot + size_t(kvh) * cap * HD);
            const int s_beg = part * per, s_end = s_beg + per < L ? s_beg + per : L;
            es::simd<uint32_t, 128> acc = 0;
            for (int s0 = s_beg; s0 < s_end; s0 += 16) {
                if constexpr (KEYMAJOR) {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        es::simd<uint32_t, 256> t = es::load_2d<uint32_t, 16, 16>(kh, HD - 1, cap - 1, HD - 1, 16 * c, s0);
                        acc ^= t.template select<128, 1>(0) ^ t.template select<128, 1>(128);
                    }
                } else {
                    #pragma unroll
                    for (int dc = 0; dc < HD / 32; ++dc)
                        acc ^= es::load_2d<uint32_t, 4, 32>(kh, cap - 1, HD - 1, cap - 1, s0 / 4, 32 * dc);
                }
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    es::simd<uint32_t, 256> t = es::load_2d<uint32_t, 16, 16>(vh, HD - 1, cap - 1, HD - 1, 16 * c, s0);
                    acc ^= t.template select<128, 1>(0) ^ t.template select<128, 1>(128);
                }
            }
            es::block_store<uint32_t, 128>(sink + size_t(it.get_global_id(0)) * 128, acc);
        });
    };
    go(); q.wait();
    const int N = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) go();
    q.wait();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
}

int main(int argc, char** argv) {
    const int L = 61440, KVH = 4, HD = 256, USERS = 4;
    const int cap = argc > 1 ? std::atoi(argv[1]) : 61440;
    std::printf("cap (row pitch) %d\n", cap);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    const size_t bytes = size_t(USERS) * KVH * HD * cap;
    uint8_t* k = sycl::malloc_device<uint8_t>(bytes, q);
    uint8_t* v = sycl::malloc_device<uint8_t>(bytes, q);
    uint32_t* sink = sycl::malloc_device<uint32_t>(size_t(USERS) * KVH * 1024 * 128, q);
    q.memset(k, 1, bytes); q.memset(v, 2, bytes); q.wait();
    std::printf("%-6s %-7s | %10s %8s | %10s %8s\n", "users", "splits", "dim-major", "GB/s", "key-major", "GB/s");
    for (int users : {1, 4})
        for (int splits : {32, 61, 128}) {
            const double a = run<false>(q, k, v, sink, users, L, cap, splits);
            const double b = run<true>(q, k, v, sink, users, L, cap, splits);
            const double gb = double(users) * KVH * L * HD * 2;
            std::printf("%-6d %-7d | %8.1f us %8.1f | %8.1f us %8.1f\n", users, splits, a, gb / a / 1e3, b, gb / b / 1e3);
        }
    return 0;
}
