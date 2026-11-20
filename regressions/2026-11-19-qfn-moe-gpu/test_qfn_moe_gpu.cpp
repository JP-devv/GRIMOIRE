// GRIMOIRE
// SPDX-License-Identifier: GPL-3.0-or-later
//
// =====================================================================
//  regressions/2026-11-19-qfn-moe-gpu  --  synthetic GPU regression
//
//  The QFN prompt MoE has three implementations that must agree:
//
//    1. baseline   launch_moe_gate_up_batched / launch_moe_down_batched
//                  (one launch per matrix over every routed slot, decode
//                  layout, float h)
//    2. XMX        the GRIMOIRE_QFN_MOE_XMX shape in src/grimoire.cpp:
//                  make_qfn_moe_route_plan -> permute_rows_bf16 ->
//                  one launch_gemm_xmx per NONEMPTY expert -> swiglu ->
//                  f32_to_bf16 -> one launch_gemm_xmx per expert ->
//                  moe_unpermute, with bf16 h
//    3. host       an independent double reference
//
//  Shape is the one the XMX branch is gated on: QFN's 512 experts,
//  H 2560, I 640, top-k 10, INT4 g128.
//
//  The weights are deliberately SPARSE and exact: every 128-wide group of
//  every row has exactly ONE nonzero column, and all scales are powers of
//  two held exactly in bf16, so the double reference visits only those
//  explicit columns (no O(K) walk, no reading of the packed arrays) and
//  the value it computes is the value the kernel's dequantizer produces.
//  What is left is the fp32 accumulation order and, for the XMX path, the
//  bf16 rounding of the SwiGLU result -- which is what the tolerances
//  below are sized for.
//
//  Routes are skewed and include the cases that break symmetric code:
//  an ALL-EMPTY token (every slot -1), tokens routed only through slots 8
//  and 9, a token whose 10 slots are all expert 0, both very low (0,1,3)
//  and very high (509,510,511) expert IDs, and one expert group big enough
//  (> 32 rows) that launch_gemm_xmx takes the large-M GEMM while another
//  stays on the DPAS tile path.  Every packed row is written at its true
//  expert offset inside the full [E*2I][H] / [E*H][I] tensors.
//
//  Two INT4 zero-point layouts are covered, per tensor:
//    affine  a real per-group zero point, value = (code - zero) * scale
//    signed  zeros == 0xff, value = s4(code) * scale
//
//  --host-only runs the fixture, the route plan and the packer round trip
//  with NO queue and NO device memory, so it is safe on a box whose GPUs
//  are busy.  The device part is a separate run.
// =====================================================================
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "kernels.hpp"            // launch_* declarations (src/kernels.hpp)
#include "b70/formats.hpp"
#include "b70/weights.hpp"
#include "b70/moe.hpp"
#include "b70/native_model.hpp"   // slice_quant_rows
#include "b70/qfn_moe_routes.hpp"
#include "b70/moe_route_remap.hpp"

namespace {

// ---- the shape the XMX branch in src/grimoire.cpp is gated on ----------
constexpr int kHidden  = 2560;   // H
constexpr int kInter   = 640;    // I
constexpr int kExperts = 512;    // E
constexpr int kTopK    = 10;     // TK
constexpr int kTokens  = 32;     // M (the branch wants SUB >= 32)
constexpr int kGroup   = 128;    // INT4 group size (g128)

constexpr int kGuRows    = 2 * kInter;          // gate rows then up rows
constexpr int kGuScales  = kHidden / kGroup;    // 20 groups per gate/up row
constexpr int kDnScales  = kInter / kGroup;     // 5 groups per down row

// Exact BF16 scales whose products with INT4 codes can need BF16 rounding.
constexpr float kGuScale = 197.0f / 8192.0f;
constexpr float kDnScale = 187.0f / 32768.0f;

// Low and high expert IDs.  Expert 0 carries the skew, 509 is the sparse one.
const int kActive[] = {0, 1, 3, 509, 510, 511};
constexpr int kNumActive = int(sizeof(kActive) / sizeof(kActive[0]));

// ---------------------------------------------------------------------
//  Deterministic fixture
// ---------------------------------------------------------------------
uint32_t mix(uint32_t a) {
    a ^= a >> 16; a *= 0x7feb352du;
    a ^= a >> 15; a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

// One explicit column per 128-wide group.  `row` is the GLOBAL row index in
// the packed tensor, so the packer and the reference cannot drift.
struct SparseEntry {
    int32_t kcol;    // column inside its group
    int32_t code;    // 4-bit code as packed
    int32_t zero;    // 0..15, or 0xff for the signed layout
};

SparseEntry sparse_entry(bool sgn, int tensor, int64_t row, int g) {
    const uint32_t seed = mix(uint32_t(uint64_t(row) * 2654435761u) ^
                              uint32_t(tensor * 2246822519u) ^
                              uint32_t(g * 3266489917u));
    const uint32_t h2 = mix(seed ^ 0x9e3779b9u);
    const uint32_t h3 = mix(seed ^ 0x85ebca6bu);
    SparseEntry s;
    s.kcol = int32_t(seed % kGroup);
    if (sgn) {
        s.code = 1 + int32_t(h2 % 15);   // 1..15: never the s4 zero
        s.zero = 0xff;
    } else {
        s.code = int32_t(h2 % 16);
        s.zero = int32_t(h3 % 16);       // a different zero point per group
        if (s.code == s.zero) s.code = (s.code + 1) & 15;   // keep it nonzero
    }
    return s;
}

// The value the kernel's dequantizer produces for that entry (b70/formats.hpp
// decode_int4): affine (code - zero) * scale, signed s4(code) * scale.
double entry_value(const SparseEntry& s, float scale) {
    if (s.zero == 0xff) {
        const int q = s.code;
        return double((q & 8) ? q - 16 : q) * double(scale);
    }
    return double(s.code - s.zero) * double(scale);
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;                  // round to nearest even
    return uint16_t(u >> 16);
}

float round_bf16(float f) {
    const uint32_t u = uint32_t(bf16_bits(f)) << 16;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct ExpertBlock {
    std::vector<uint8_t>  payload;   // [rows][row_bytes]
    std::vector<uint16_t> scales;    // [rows][row_scales], bf16 bits
    std::vector<uint8_t>  zeros;     // [rows][row_scales]
};

// Fill one expert's rows at `first_row` of the packed tensor.
void build_block(ExpertBlock& blk, bool sgn, int tensor, int64_t first_row,
                 int rows, int row_scales, int row_bytes, float scale) {
    blk.payload.assign(size_t(rows) * size_t(row_bytes), 0);
    blk.scales.assign(size_t(rows) * size_t(row_scales), bf16_bits(scale));
    blk.zeros.assign(size_t(rows) * size_t(row_scales), sgn ? uint8_t(0xff) : uint8_t(0));
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = blk.payload.data() + size_t(r) * size_t(row_bytes);
        for (int g = 0; g < row_scales; ++g) {
            const SparseEntry s = sparse_entry(sgn, tensor, first_row + r, g);
            uint8_t* group = row + g * (kGroup / 2);
            if (!sgn) {
                // An affine zero has code `zero`, not code 0.
                std::fill_n(group, kGroup / 2, uint8_t(s.zero * 0x11));
                blk.zeros[size_t(r) * row_scales + g] = uint8_t(s.zero);
            }
            uint8_t& byte = group[s.kcol >> 1];
            const uint8_t nib = uint8_t(s.code);
            byte = (s.kcol & 1) ? uint8_t((byte & 0x0f) | uint8_t(nib << 4))
                                : uint8_t((byte & 0xf0) | nib);
        }
    }
}

struct Blocks {
    std::vector<ExpertBlock> gu;   // one per kActive entry
    std::vector<ExpertBlock> dn;
};

void build_blocks(Blocks& b, bool sgn_gu, bool sgn_dn) {
    const int gu_rb = b70::bytes_per_row(b70::Fmt::INT4, kHidden);
    const int dn_rb = b70::bytes_per_row(b70::Fmt::INT4, kInter);
    b.gu.resize(kNumActive);
    b.dn.resize(kNumActive);
    for (int a = 0; a < kNumActive; ++a) {
        const int e = kActive[a];
        build_block(b.gu[a], sgn_gu, /*tensor=*/0, int64_t(e) * kGuRows,
                    kGuRows, kGuScales, gu_rb, kGuScale);
        build_block(b.dn[a], sgn_dn, /*tensor=*/1, int64_t(e) * kHidden,
                    kHidden, kDnScales, dn_rb, kDnScale);
    }
}

// ---------------------------------------------------------------------
//  Routes: skewed, with the four shapes that break symmetric code
// ---------------------------------------------------------------------
struct Routes {
    std::vector<int32_t> ids;      // [M][top_k], -1 = absent
    std::vector<float>   weight;   // [M][top_k]
    std::vector<int32_t> counts;   // [E]
};

Routes make_routes() {
    Routes r;
    r.ids.assign(size_t(kTokens) * kTopK, -1);
    r.weight.assign(size_t(kTokens) * kTopK, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
        int32_t* row = r.ids.data() + size_t(t) * kTopK;
        if (t == 0) {
            // all-empty token: every slot -1, must produce an exact zero row
        } else if (t % 7 == 2) {
            // only slots 8 and 9 route: low expert and high expert
            row[8] = 0;
            row[9] = 511;
        } else if (t % 5 == 3) {
            // maximum skew: all ten slots on expert 0
            for (int s = 0; s < kTopK; ++s) row[s] = 0;
        } else {
            row[0] = 0;
            if (t % 2 == 0)  row[1] = 1;
            if (t % 3 == 0)  row[2] = 3;
            if (t % 5 == 0)  row[3] = 511;
            if (t % 7 == 0)  row[4] = 510;
            if (t % 11 == 0) row[5] = 509;
            if (t % 4 == 0)  row[6] = 511;   // low and high in one token
        }
        for (int s = 0; s < kTopK; ++s)
            r.weight[size_t(t) * kTopK + s] = std::ldexp(1.0f, -((t + s) % 3));
    }
    r.counts.assign(size_t(kExperts), 0);
    for (int32_t id : r.ids) if (id >= 0) ++r.counts[size_t(id)];
    return r;
}

// ---------------------------------------------------------------------
//  Independent double reference
//
//  y[t][h] = sum over slots s with e = ids[t][s] >= 0 of
//              w[t][s] * sum_groups v_down(e,h,g) * sw[t][s][i_g]
//  with sw[t][s][i] = silu(gate_e(x_t)[i]) * up_e(x_t)[i] and, for every
//  row, the ONLY terms that exist are the explicit ones the fixture wrote.
// ---------------------------------------------------------------------
void reference_forward(const Routes& rt, const std::vector<float>& x,
                       bool sgn_gu, bool sgn_dn, bool use_bf16, std::vector<double>& y) {
    const auto operand = [use_bf16](double v) -> double {
        return use_bf16 ? round_bf16(float(v)) : v;
    };
    y.assign(size_t(kTokens) * kHidden, 0.0);
    std::vector<double> sw(static_cast<size_t>(kInter));
    for (int t = 0; t < kTokens; ++t) {
        double* out = y.data() + size_t(t) * kHidden;
        const float* xt = x.data() + size_t(t) * kHidden;
        for (int s = 0; s < kTopK; ++s) {
            const int e = rt.ids[size_t(t) * kTopK + s];
            if (e < 0) continue;
            const double w = double(rt.weight[size_t(t) * kTopK + s]);
            if (w == 0.0) continue;
            const int64_t gu_base = int64_t(e) * kGuRows;
            const int64_t dn_base = int64_t(e) * kHidden;
            for (int i = 0; i < kInter; ++i) {
                double g = 0.0, u = 0.0;
                for (int grp = 0; grp < kGuScales; ++grp) {
                    const SparseEntry sg = sparse_entry(sgn_gu, 0, gu_base + i, grp);
                    const SparseEntry su = sparse_entry(sgn_gu, 0, gu_base + kInter + i, grp);
                    g += operand(entry_value(sg, kGuScale)) * operand(xt[sg.kcol + grp * kGroup]);
                    u += operand(entry_value(su, kGuScale)) * operand(xt[su.kcol + grp * kGroup]);
                }
                sw[size_t(i)] = operand((g / (1.0 + std::exp(-g))) * u);
            }
            for (int h = 0; h < kHidden; ++h) {
                double acc = 0.0;
                for (int grp = 0; grp < kDnScales; ++grp) {
                    const SparseEntry sd = sparse_entry(sgn_dn, 1, dn_base + h, grp);
                    acc += operand(entry_value(sd, kDnScale)) * sw[size_t(sd.kcol + grp * kGroup)];
                }
                out[h] += w * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------
//  Comparison
// ---------------------------------------------------------------------
struct Diff {
    double  max_abs = 0.0;
    double  max_rel = 0.0;
    int64_t at = -1;
    double  got = 0.0, want = 0.0;
};

Diff diff_ref(const std::vector<double>& ref, const float* got, int64_t n) {
    Diff d;
    for (int64_t i = 0; i < n; ++i) {
        const double a = double(got[i]), b = ref[size_t(i)];
        if (!std::isfinite(a) || !std::isfinite(b)) return {INFINITY, INFINITY, i, a, b};
        const double ea = std::fabs(a - b);
        const double er = ea / std::fmax(std::fabs(b), 1e-30);
        if (ea > d.max_abs) d.max_abs = ea;
        if (er > d.max_rel) { d.max_rel = er; d.at = i; d.got = a; d.want = b; }
    }
    return d;
}

Diff diff_ff(const float* a, const float* b, int64_t n) {
    Diff d;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return {INFINITY, INFINITY, i, a[i], b[i]};
        const double ea = std::fabs(double(a[i]) - double(b[i]));
        const double er = ea / std::fmax(std::fabs(double(b[i])), 1e-30);
        if (ea > d.max_abs) d.max_abs = ea;
        if (er > d.max_rel) { d.max_rel = er; d.at = i; d.got = a[i]; d.want = b[i]; }
    }
    return d;
}

bool report(const char* what, const Diff& d, double rel_tol, double abs_tol,
            bool informational = false) {
    const bool ok = (d.max_rel <= rel_tol) || (d.max_abs <= abs_tol);
    std::printf("    %-22s max_abs %-11.4g max_rel %-11.4g tol(rel %.0e/abs %.0e) %s",
                what, d.max_abs, d.max_rel, rel_tol, abs_tol,
                informational ? (ok ? "ok" : "INFO") : (ok ? "PASS" : "FAIL"));
    if (!ok && d.at >= 0)
        std::printf("  worst idx %lld got %.9g want %.9g",
                    (long long)d.at, d.got, d.want);
    std::printf("\n");
    return informational ? true : ok;
}

// ---------------------------------------------------------------------
//  Host-only checks: routes, plan, packer round trip.  No device touched.
// ---------------------------------------------------------------------
bool host_checks(const Routes& rt, bool verbose) {
    bool ok = true;
    std::printf("  routes: %d tokens x top-k %d = %zu slots, %zu active\n",
                kTokens, kTopK, rt.ids.size(),
                size_t(std::count_if(rt.ids.begin(), rt.ids.end(),
                                     [](int32_t v) { return v >= 0; })));
    for (int a = 0; a < kNumActive; ++a)
        std::printf("    expert %3d : %d rows\n", kActive[a], rt.counts[size_t(kActive[a])]);
    for (int a = 0; a < kNumActive; ++a)
        if (rt.counts[size_t(kActive[a])] <= 0) {
            std::printf("    FAIL: expert %d is active in kActive but never routed\n", kActive[a]);
            ok = false;
        }
    bool big = false, small = false;
    for (int a = 0; a < kNumActive; ++a) {
        if (rt.counts[size_t(kActive[a])] >= 32) big = true;
        if (rt.counts[size_t(kActive[a])] < 32) small = true;
    }
    std::printf("    both GEMM paths exercised: >=32 rows %s, <32 rows %s\n",
                big ? "yes" : "NO", small ? "yes" : "NO");
    ok = ok && big && small;

    // the all-empty token stays all -1
    for (int s = 0; s < kTopK; ++s)
        if (rt.ids[size_t(s)] != -1) {
            std::printf("    FAIL: token 0 is not all-empty\n");
            ok = false;
        }

    // route plan: offsets, inverse, perm_token
    const auto plan = b70::make_qfn_moe_route_plan(rt.ids.data(), rt.ids.size(),
                                                   kTokens, kTopK, kExperts);
    int64_t total = 0;
    for (int e = 0; e < kExperts; ++e) {
        if (plan.counts[size_t(e)] != rt.counts[size_t(e)]) {
            std::printf("    FAIL: plan count for expert %d is %d, expected %d\n",
                        e, plan.counts[size_t(e)], rt.counts[size_t(e)]);
            ok = false;
        }
        total += plan.counts[size_t(e)];
        if (plan.offsets[size_t(e)] != int32_t(total - plan.counts[size_t(e)])) {
            std::printf("    FAIL: plan offset %d for expert %d\n", plan.offsets[size_t(e)], e);
            ok = false;
        }
    }
    if (plan.offsets.back() != total) {
        std::printf("    FAIL: offsets.back() %d != active rows %lld\n",
                    plan.offsets.back(), (long long)total);
        ok = false;
    }
    for (size_t r = 0; r < rt.ids.size(); ++r) {
        const int32_t p = plan.inverse[r];
        if (rt.ids[r] < 0) {
            if (p != -1) { std::printf("    FAIL: inverse[%zu] %d for an absent slot\n", r, p); ok = false; }
        } else if (p < 0 || plan.perm_token[size_t(p)] != int32_t(r / kTopK)) {
            std::printf("    FAIL: inverse[%zu] %d does not map back to its token\n", r, p);
            ok = false;
        }
    }
    std::printf("  plan: %lld compact rows, offsets monotone, inverse/perm_token consistent\n",
                (long long)total);

    // packer round trip: decode the packed row with the library's own
    // dequantizer and check every nonzero lands on the chosen column with
    // the chosen value, and that there are exactly row_scales of them.
    Blocks blocks;
    for (int layout = 0; layout < 2; ++layout) {
        const bool sgn = layout == 1;
        build_blocks(blocks, sgn, sgn);
        for (int a = 0; a < kNumActive; ++a) {
            for (int which = 0; which < 2; ++which) {
                const int rows = which == 0 ? kGuRows : kHidden;
                const int K = which == 0 ? kHidden : kInter;
                const int row_scales = which == 0 ? kGuScales : kDnScales;
                const float scale = which == 0 ? kGuScale : kDnScale;
                const int tensor = which;
                const int64_t base = int64_t(kActive[a]) * (which == 0 ? kGuRows : kHidden);
                const ExpertBlock& blk = which == 0 ? blocks.gu[size_t(a)] : blocks.dn[size_t(a)];
                b70::QuantWeight w;
                w.fmt = b70::Fmt::INT4;
                w.N = rows;
                w.K = K;
                w.payload = blk.payload.data();
                w.scales = blk.scales.data();
                w.zeros = blk.zeros.data();
                w.row_bytes = int64_t(blk.payload.size()) / rows;
                w.row_scales = row_scales;
                for (int r = 0; r < rows; r += rows / 2) {   // two rows is enough
                    int nonzero = 0;
                    for (int k = 0; k < K; ++k) {
                        const double got = double(w.at(r, k));
                        const int g = k / kGroup;
                        const SparseEntry s = sparse_entry(sgn, tensor, base + r, g);
                        const bool is_pick = (k % kGroup) == s.kcol;
                        const double want = is_pick ? entry_value(s, scale) : 0.0;
                        if (got == want && got != 0.0) ++nonzero;
                        else if (got != want) {
                            std::printf("    FAIL: %s expert %d row %d k %d got %.9g want %.9g\n",
                                        which == 0 ? "gate_up" : "down", kActive[a], r, k,
                                        got, want);
                            ok = false;
                            goto next_block;
                        }
                        (void)verbose;
                    }
                    if (nonzero != row_scales) {
                        std::printf("    FAIL: %s expert %d row %d has %d nonzeros, expected %d\n",
                                    which == 0 ? "gate_up" : "down", kActive[a], r,
                                    nonzero, row_scales);
                        ok = false;
                    }
                }
            next_block:;
            }
        }
        std::printf("  packer: %s layout round-trips through QuantWeight::at\n",
                    sgn ? "signed 0xff" : "affine");
    }
    return ok;
}

// Same in-order remap and resident GEMV dispatch as QFN TP prefill. The
// whole-table control retains global IDs and masks only the route IDs.
bool check_rank1_gemv(sycl::queue& q, const b70::MoeLayer& full,
                      int32_t* ids, float* weights, const float* x, float* h,
                      float* full_y, float* local_y) {
    constexpr int begin = 128, end = 256, local_count = end - begin;
    static_assert(b70::moe_routes_need_remap(local_count, kExperts));
    static_assert(!b70::moe_routes_need_remap(kExperts, kExperts));
    constexpr int tokens = 3;
    std::array<int32_t, tokens * kTopK> global;
    global.fill(-1);
    const std::array<int32_t, 4> boundary{127, 128, 255, 256};
    std::copy(boundary.begin(), boundary.end(), global.begin());
    global[kTopK] = 127;               // no expert local to rank 1
    global[kTopK + 1] = 256;
    global[2 * kTopK + 8] = 255;       // route slots beyond the first eight
    global[2 * kTopK + 9] = 128;
    std::array<float, tokens * kTopK> original_weights{};
    for (size_t i = 0; i < global.size(); ++i)
        if (global[i] >= 0) original_weights[i] = float(i + 1) / 64.0f;
    auto masked = global;
    for (int32_t& e : masked) if (e != 128 && e != 255) e = -1;
    auto expected = masked;
    for (int32_t& e : expected) {
        if (e == 128) e = 0;
        else if (e == 255) e = 127;
    }

    b70::MoeLayer local = full;
    local.cfg.num_experts = local_count;
    std::vector<void*> owned;
    auto allocate = [&](size_t n) {
        auto* p = sycl::malloc_device<uint8_t>(n, q);
        if (!p) throw std::bad_alloc();
        owned.push_back(p);
        q.memset(p, 0, n);
        return p;
    };
    const auto pack = [&](b70::QuantWeight& w, int rows_per_expert, int tensor, float scale) {
        w.N = local_count * rows_per_expert;
        auto* payload = allocate(size_t(w.N) * w.row_bytes);
        auto* scales = allocate(size_t(w.N) * w.row_scales * sizeof(uint16_t));
        auto* zeros = allocate(size_t(w.N) * w.row_scales);
        const auto& ref = tensor == 0 ? full.gate_up : full.down;
        for (int e : boundary) {
            ExpertBlock block;
            build_block(block, true, tensor, int64_t(e) * rows_per_expert, rows_per_expert,
                        w.row_scales, int(w.row_bytes), scale);
            const auto upload = [&](const uint8_t* p, const void* s, const uint8_t* z, size_t row) {
                q.memcpy(const_cast<uint8_t*>(p) + row * w.row_bytes,
                         block.payload.data(), block.payload.size());
                q.memcpy(const_cast<uint8_t*>(static_cast<const uint8_t*>(s)) +
                         row * w.row_scales * sizeof(uint16_t),
                         block.scales.data(), block.scales.size() * sizeof(uint16_t));
                q.memcpy(const_cast<uint8_t*>(z) + row * w.row_scales,
                         block.zeros.data(), block.zeros.size());
            };
            upload(ref.payload, ref.scales, ref.zeros, size_t(e) * rows_per_expert);
            if (e == 128 || e == 255) {
                const size_t slot = e == 128 ? 0 : 127;
                upload(payload, scales, zeros, slot * rows_per_expert);
            }
            q.wait_and_throw();  // the host block is reused next iteration
        }
        w.payload = payload;
        w.scales = scales;
        w.zeros = zeros;
    };
    pack(local.gate_up, kGuRows, 0, kGuScale);
    pack(local.down, kHidden, 1, kDnScale);
    q.memcpy(weights, original_weights.data(), sizeof(original_weights));

    const auto dispatch = [&](const b70::MoeLayer& layer, int rows, float* out) {
        q.memset(h, 0xFF, size_t(rows) * kTopK * kInter * sizeof(float));
        q.memset(out, 0xFF, size_t(rows) * kHidden * sizeof(float));
        b70::launch_moe_gate_up_batched(q, layer, ids, x, h, rows);
        b70::launch_moe_down_batched(q, layer, ids, weights, h, out, rows);
    };

    bool ok = true;
    std::vector<float> actual(tokens * kHidden), reference(actual.size());
    for (int rows : {1, tokens}) {
        const size_t route_bytes = size_t(rows) * kTopK * sizeof(int32_t);
        const size_t elems = size_t(rows) * kHidden;
        q.memcpy(ids, masked.data(), route_bytes);
        dispatch(full, rows, full_y);
        q.memcpy(reference.data(), full_y, elems * sizeof(float)).wait_and_throw();

        q.memcpy(ids, global.data(), route_bytes);
        if (b70::moe_routes_need_remap(local.cfg.num_experts, full.cfg.num_experts))
            b70::launch_remap_moe_routes(q, ids, size_t(rows) * kTopK, begin, end);
        std::array<int32_t, tokens * kTopK> remapped{};
        q.memcpy(remapped.data(), ids, route_bytes).wait_and_throw();
        std::printf("  rank 1 rows=%d: [127, 128, 255, 256] -> [%d, %d, %d, %d]\n",
                    rows, remapped[0], remapped[1], remapped[2], remapped[3]);
        if (!std::equal(remapped.begin(), remapped.begin() + rows * kTopK, expected.begin())) {
            ok = false;
            std::printf("    FAIL: rank-local route ids\n");
            break;  // never feed an invalid expert ID to GEMV
        }
        dispatch(local, rows, local_y);
        q.memcpy(actual.data(), local_y, elems * sizeof(float)).wait_and_throw();
        ok &= report("rank 1 vs masked full", diff_ff(actual.data(), reference.data(), elems), 0, 0);
        if (rows == tokens) {
            const bool zero = std::all_of(actual.begin() + kHidden, actual.begin() + 2 * kHidden,
                                          [](float v) { return v == 0; });
            std::printf("    no-local-route row: %s\n", zero ? "PASS" : "FAIL");
            ok &= zero;
        }
        if (rows == 1) {
            std::swap(remapped[1], remapped[2]);
            q.memcpy(ids, remapped.data(), route_bytes);
            dispatch(local, rows, local_y);
            q.memcpy(actual.data(), local_y, elems * sizeof(float)).wait_and_throw();
            const auto swapped = diff_ff(actual.data(), reference.data(), elems);
            const bool distinct = std::isfinite(swapped.max_abs) && swapped.max_abs > 0;
            std::printf("    swapped endpoint experts: max_abs %.6g %s\n",
                        swapped.max_abs, distinct ? "PASS (different)" : "FAIL");
            ok &= distinct;
        }
    }
    for (auto* p : owned) sycl::free(p, q);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const bool host_only = argc > 1 && std::strcmp(argv[1], "--host-only") == 0;
    // The normal per-expert dispatcher can choose oneDNN A8 instead of
    // BF16 DPAS. Disable it before the first dispatch for an exact comparison.
    bool force_dpas = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--force-dpas") == 0) force_dpas = true;
    if (force_dpas) setenv("GRIMOIRE_NO_FAST_GEMM", "1", 1);

    std::printf("GRIMOIRE QFN MoE GPU regression\n");
    std::printf("  shape  M=%d H=%d I=%d E=%d top-k=%d INT4 g%d\n",
                kTokens, kHidden, kInter, kExperts, kTopK, kGroup);

    const Routes rt = make_routes();
    if (!host_checks(rt, true)) {
        std::printf("HOST CHECKS FAILED\n");
        return 1;
    }
    if (host_only) {
        std::printf("host-only: OK (no queue created, no device memory touched)\n");
        return 0;
    }

    // ---- dense varying activations, including BF16 rounding cases ------
    std::vector<float> x(size_t(kTokens) * kHidden);
    for (int t = 0; t < kTokens; ++t)
        for (int k = 0; k < kHidden; ++k)
            x[size_t(t) * kHidden + k] = float(int(mix(uint32_t(t * kHidden + k)) % 2003) - 1001) / 509.0f;

    bool ok = true;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("  device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    if (!q.is_in_order()) {
        std::printf("FAIL: the queue is not in-order; the XMX sequence relies on it\n");
        return 1;
    }

    // ---- buffers ------------------------------------------------------
    const int gu_rb = b70::bytes_per_row(b70::Fmt::INT4, kHidden);
    const int dn_rb = b70::bytes_per_row(b70::Fmt::INT4, kInter);
    const int64_t gu_rows = int64_t(kExperts) * kGuRows;
    const int64_t dn_rows = int64_t(kExperts) * kHidden;
    const size_t slots = size_t(kTokens) * kTopK;
    const int rows_max = kTokens * kTopK;

    uint8_t*  d_gu_p = sycl::malloc_device<uint8_t>(size_t(gu_rows) * gu_rb, q);
    uint16_t* d_gu_s = sycl::malloc_device<uint16_t>(size_t(gu_rows) * kGuScales, q);
    uint8_t*  d_gu_z = sycl::malloc_device<uint8_t>(size_t(gu_rows) * kGuScales, q);
    uint8_t*  d_dn_p = sycl::malloc_device<uint8_t>(size_t(dn_rows) * dn_rb, q);
    uint16_t* d_dn_s = sycl::malloc_device<uint16_t>(size_t(dn_rows) * kDnScales, q);
    uint8_t*  d_dn_z = sycl::malloc_device<uint8_t>(size_t(dn_rows) * kDnScales, q);
    float*    d_x    = sycl::malloc_device<float>(size_t(kTokens) * kHidden, q);
    int32_t*  d_rex  = sycl::malloc_device<int32_t>(slots, q);
    float*    d_rwt  = sycl::malloc_device<float>(slots, q);
    float*    d_mh   = sycl::malloc_device<float>(size_t(kTokens) * kTopK * kInter, q);
    float*    d_yb   = sycl::malloc_device<float>(size_t(kTokens) * kHidden, q);
    float*    d_yx   = sycl::malloc_device<float>(size_t(kTokens) * kHidden, q);
    b70::sycl_bf16* xmx_x  = sycl::malloc_device<b70::sycl_bf16>(size_t(rows_max) * kHidden, q);
    float*    xmx_gu = sycl::malloc_device<float>(size_t(rows_max) * 2 * kInter, q);
    float*    xmx_y  = sycl::malloc_device<float>(size_t(rows_max) * kHidden, q);
    int32_t*  d_pt   = sycl::malloc_device<int32_t>(slots, q);
    int32_t*  d_inv  = sycl::malloc_device<int32_t>(slots, q);
    auto* d_tiles = sycl::malloc_device<b70::QfnMoeTile>(slots, q);

    if (!d_gu_p || !d_gu_s || !d_gu_z || !d_dn_p || !d_dn_s || !d_dn_z || !d_x || !d_rex ||
        !d_rwt || !d_mh || !d_yb || !d_yx || !xmx_x || !xmx_gu || !xmx_y || !d_pt || !d_inv || !d_tiles) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    std::printf("  device memory: %.2f GiB of routed weights\n",
                double(size_t(gu_rows) * gu_rb + size_t(dn_rows) * dn_rb +
                       size_t(gu_rows) * kGuScales * 3 + size_t(dn_rows) * kDnScales * 3) /
                    1073741824.0);

    // inactive experts are never read; the memset only keeps the fixture honest
    q.memset(d_gu_p, 0, size_t(gu_rows) * gu_rb);
    q.memset(d_gu_s, 0, size_t(gu_rows) * kGuScales * 2);
    q.memset(d_gu_z, 0, size_t(gu_rows) * kGuScales);
    q.memset(d_dn_p, 0, size_t(dn_rows) * dn_rb);
    q.memset(d_dn_s, 0, size_t(dn_rows) * kDnScales * 2);
    q.memset(d_dn_z, 0, size_t(dn_rows) * kDnScales);
    q.memcpy(d_x, x.data(), size_t(kTokens) * kHidden * sizeof(float));
    q.memcpy(d_rex, rt.ids.data(), slots * sizeof(int32_t));
    q.memcpy(d_rwt, rt.weight.data(), slots * sizeof(float));
    q.wait_and_throw();

    b70::MoeLayer L;
    L.cfg.hidden = kHidden;
    L.cfg.inter = kInter;
    L.cfg.num_experts = kExperts;
    L.cfg.top_k = kTopK;
    L.cfg.shared_inter = 0;
    L.cfg.norm_topk = true;
    L.gate_up.fmt = b70::Fmt::INT4;
    L.gate_up.N = int(gu_rows);
    L.gate_up.K = kHidden;
    L.gate_up.payload = d_gu_p;
    L.gate_up.scales = d_gu_s;
    L.gate_up.zeros = d_gu_z;
    L.gate_up.row_bytes = gu_rb;
    L.gate_up.row_scales = kGuScales;
    L.down = L.gate_up;
    L.down.N = int(dn_rows);
    L.down.K = kInter;
    L.down.payload = d_dn_p;
    L.down.scales = d_dn_s;
    L.down.zeros = d_dn_z;
    L.down.row_bytes = dn_rb;
    L.down.row_scales = kDnScales;
    L.has_shared = false;

    struct Case { const char* name; bool sgn_gu; bool sgn_dn; };
    const Case cases[] = {
        {"affine gu + affine down", false, false},
        {"signed gu + signed down", true,  true},
        {"affine gu + signed down", false, true},
    };

    std::vector<double> ref, bf_ref;
    std::vector<float> yb(size_t(kTokens) * kHidden), yx(yb.size()), yg(yb.size());
    std::vector<int32_t> ids_back(slots);
    Blocks blocks;

    for (const Case& c : cases) {
        std::printf("\n[%s]\n", c.name);
        build_blocks(blocks, c.sgn_gu, c.sgn_dn);
        for (int a = 0; a < kNumActive; ++a) {
            const int e = kActive[a];
            q.memcpy(d_gu_p + int64_t(e) * kGuRows * gu_rb,
                     blocks.gu[size_t(a)].payload.data(), size_t(kGuRows) * gu_rb);
            q.memcpy(d_gu_s + int64_t(e) * kGuRows * kGuScales,
                     blocks.gu[size_t(a)].scales.data(), size_t(kGuRows) * kGuScales * 2);
            q.memcpy(d_gu_z + int64_t(e) * kGuRows * kGuScales,
                     blocks.gu[size_t(a)].zeros.data(), size_t(kGuRows) * kGuScales);
            q.memcpy(d_dn_p + int64_t(e) * kHidden * dn_rb,
                     blocks.dn[size_t(a)].payload.data(), size_t(kHidden) * dn_rb);
            q.memcpy(d_dn_s + int64_t(e) * kHidden * kDnScales,
                     blocks.dn[size_t(a)].scales.data(), size_t(kHidden) * kDnScales * 2);
            q.memcpy(d_dn_z + int64_t(e) * kHidden * kDnScales,
                     blocks.dn[size_t(a)].zeros.data(), size_t(kHidden) * kDnScales);
        }
        q.wait_and_throw();

        // ---- 1. baseline -------------------------------------------------
        q.memset(d_mh, 0, size_t(kTokens) * kTopK * kInter * sizeof(float));
        q.memset(d_yb, 0, size_t(kTokens) * kHidden * sizeof(float));
        b70::launch_moe_gate_up_batched(q, L, d_rex, d_x, d_mh, kTokens);
        b70::launch_moe_down_batched(q, L, d_rex, d_rwt, d_mh, d_yb, kTokens);
        q.wait_and_throw();
        q.memcpy(yb.data(), d_yb, size_t(kTokens) * kHidden * sizeof(float)).wait_and_throw();

        // ---- 2. the GRIMOIRE_QFN_MOE_XMX sequence ------------------------
        q.memcpy(ids_back.data(), d_rex, slots * sizeof(int32_t)).wait_and_throw();
        const auto plan = b70::make_qfn_moe_route_plan(ids_back.data(), slots,
                                                      kTokens, kTopK, kExperts);
        const int rows = plan.offsets.back();
        const auto tiles = b70::make_qfn_moe_tiles(plan);
        q.memcpy(d_tiles, tiles.data(), tiles.size() * sizeof(b70::QfnMoeTile));
        for (int e = 0; e < kExperts; ++e)
            if (plan.counts[size_t(e)] != rt.counts[size_t(e)]) {
                std::printf("    FAIL: device routes disagree with the fixture at expert %d\n", e);
                ok = false;
            }
        q.memcpy(d_inv, plan.inverse.data(), slots * sizeof(int32_t));
        if (rows) q.memcpy(d_pt, plan.perm_token.data(), size_t(rows) * sizeof(int32_t));
        q.wait_and_throw();

        q.memset(d_mh, 0, size_t(kTokens) * kTopK * kInter * sizeof(float));
        q.memset(d_yx, 0, size_t(kTokens) * kHidden * sizeof(float));
        q.memset(xmx_y, 0, size_t(rows_max) * kHidden * sizeof(float));
        if (rows) {
            b70::launch_permute_rows_bf16(q, d_x, d_pt, xmx_x, rows, kHidden);
            for (int e = 0; e < kExperts; ++e) {
                if (!plan.counts[size_t(e)]) continue;
                const size_t off = size_t(plan.offsets[size_t(e)]);
                const auto w = b70::slice_quant_rows(L.gate_up, e * 2 * kInter, 2 * kInter);
                b70::launch_gemm_xmx(q, w, xmx_x + off * kHidden,
                                     xmx_gu + off * 2 * kInter, plan.counts[size_t(e)]);
            }
            b70::launch_swiglu_batched(q, xmx_gu, d_mh, rows, kInter);
            b70::launch_f32_to_bf16(q, d_mh, xmx_x, size_t(rows) * kInter);
            for (int e = 0; e < kExperts; ++e) {
                if (!plan.counts[size_t(e)]) continue;
                const size_t off = size_t(plan.offsets[size_t(e)]);
                const auto w = b70::slice_quant_rows(L.down, e * kHidden, kHidden);
                b70::launch_gemm_xmx(q, w, xmx_x + off * kInter,
                                     xmx_y + off * kHidden, plan.counts[size_t(e)]);
            }
        }
        b70::launch_moe_unpermute(q, xmx_y, d_inv, d_rwt, d_yx, kTokens, kTopK, kHidden);
        q.wait_and_throw();
        q.memcpy(yx.data(), d_yx, size_t(kTokens) * kHidden * sizeof(float)).wait_and_throw();

        // ---- 3. grouped: same routing and rounding, one GEMM per projection
        q.memset(xmx_gu, 0xFF, size_t(rows) * kGuRows * sizeof(float));
        q.memset(xmx_y, 0xFF, size_t(rows) * kHidden * sizeof(float));
        b70::launch_permute_rows_bf16(q, d_x, d_pt, xmx_x, rows, kHidden);
        b70::launch_qfn_int4_grouped(q, L.gate_up, kGuRows, xmx_x, xmx_gu,
                                     d_tiles, int(tiles.size()));
        b70::launch_swiglu_bf16_out(q, xmx_gu, xmx_x, rows, kInter);
        b70::launch_qfn_int4_grouped(q, L.down, kHidden, xmx_x, xmx_y,
                                     d_tiles, int(tiles.size()));
        b70::launch_moe_unpermute(q, xmx_y, d_inv, d_rwt, d_yx, kTokens, kTopK, kHidden);
        q.memcpy(yg.data(), d_yx, yg.size() * sizeof(float)).wait_and_throw();

        reference_forward(rt, x, c.sgn_gu, c.sgn_dn, false, ref);
        reference_forward(rt, x, c.sgn_gu, c.sgn_dn, true, bf_ref);
        double scale_ref = 0.0;
        for (double v : ref) scale_ref = std::fmax(scale_ref, std::fabs(v));
        std::printf("    compact rows %d, tiles %zu, |y| max %.4g\n", rows, tiles.size(), scale_ref);

        // Any expert big enough for gemm_fast took oneDNN A8, so the
        // per-expert arm is a MIXED-precision control, not a bf16 one.
        bool mixed = false;
        for (int e = 0; e < kExperts; ++e) {
            if (!plan.counts[size_t(e)]) continue;
            if (b70::gemm_fast_supported(b70::slice_quant_rows(L.gate_up, e * kGuRows, kGuRows),
                                         plan.counts[size_t(e)])) mixed = true;
        }

        ok &= report("baseline vs ref", diff_ref(ref, yb.data(), int64_t(yb.size())),
                     1e-3, 1e-5 * scale_ref);
        ok &= report("grouped vs BF16 ref", diff_ref(bf_ref, yg.data(), int64_t(yg.size())),
                     0, 3e-3 * scale_ref);
        ok &= report("grouped vs xmx", diff_ff(yg.data(), yx.data(), int64_t(yx.size())),
                     0, mixed ? 1e-2 * scale_ref : 0.0f, mixed);

        // The all-empty token must be exactly zero in every arm.
        bool zero_ok = true;
        for (int h = 0; h < kHidden; ++h)
            if (yb[size_t(h)] != 0.0f || yx[size_t(h)] != 0.0f || yg[size_t(h)] != 0.0f) zero_ok = false;
        std::printf("    all-empty token exact zero: %s\n", zero_ok ? "yes" : "NO");
        ok = ok && zero_ok;

        // per-expert GEMM route, logged because it decides how many kernels ran
        for (int a = 0; a < kNumActive; ++a) {
            const int e = kActive[a];
            if (!plan.counts[size_t(e)]) continue;
            const auto wg = b70::slice_quant_rows(L.gate_up, e * 2 * kInter, 2 * kInter);
            const auto wd = b70::slice_quant_rows(L.down, e * kHidden, kHidden);
            std::printf("    expert %3d : %3d rows, gate_up %s, down %s\n", e,
                        plan.counts[size_t(e)],
                        b70::gemm_fast_supported(wg, plan.counts[size_t(e)]) ? "large-M" : "dpas",
                        b70::gemm_fast_supported(wd, plan.counts[size_t(e)]) ? "large-M" : "dpas");
        }
    }

    std::printf("\n[TP4 rank 1 resident GEMV]\n");
    ok &= check_rank1_gemv(q, L, d_rex, d_rwt, d_x, d_mh, d_yb, d_yx);

    sycl::free(d_gu_p, q); sycl::free(d_gu_s, q); sycl::free(d_gu_z, q);
    sycl::free(d_dn_p, q); sycl::free(d_dn_s, q); sycl::free(d_dn_z, q);
    sycl::free(d_x, q);    sycl::free(d_rex, q);  sycl::free(d_rwt, q);
    sycl::free(d_mh, q);   sycl::free(d_yb, q);   sycl::free(d_yx, q);
    sycl::free(xmx_x, q);  sycl::free(xmx_gu, q); sycl::free(xmx_y, q);
    sycl::free(d_pt, q);   sycl::free(d_inv, q); sycl::free(d_tiles, q);

    std::printf("\n%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
