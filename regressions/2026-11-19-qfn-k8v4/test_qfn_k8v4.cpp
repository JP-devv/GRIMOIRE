// SPDX-License-Identifier: GPL-3.0-or-later
//
// K8/V4 scalar reference fixture -- D=256, GQA=6, ONE KV head shared by six
// query heads (a TP4-shaped per-rank slice). Host only: no SYCL, no GPU, no
// vLLM, no checkpoint.
//
// DOCUMENTED (read from the staged reference tree, nothing executed from it):
//   qfn-enhanced/k8v4.md, section 2:
//     PAGE=64, D=256, GQA=6, per head per token 256 B int8 K + 128 B packed
//     int4 V + 3*4 B fp32 scales = 396 B; region-major page order
//     K,V,k_scale,v_scale,v_zero; PageLayout accepts hkv in {2,4} only and
//     for_heads() demands num_heads == 6*num_kv_heads.
//   k8v4_v030/layout.py: the same constants; region_view_specs strides
//     (page_bytes, hkv*D, D, 1) for K with page stride leaving the gap the
//     other regions occupy; element_byte() walks those strides.
//   k8v4_v030/oracle.py: K symmetric int8, scale = amax/127, round half away
//     from zero, clamp +-127; V affine int4, scale = max((hi-lo)/15, 1e-8),
//     zero = clamp(floor(-lo/scale + 0.5), 0, 15), nibble =
//     clamp(floor(v/scale + zero + 0.5), 0, 15), low nibble = even dimension;
//     scores = int8 dot * q_scale * k_scale / sqrt(256).
//   k8v4_v030/native/xe2_kv_ops.cpp (read, not built):
//     * the V store uses exactly the rule above (byte = q_even | q_odd << 4);
//     * K/Q quantize with sycl::rint, i.e. round-to-nearest TIES-TO-EVEN, not
//       the documented round-half-away -- see test_k_quant() for the
//       constructed tie row where the two rules disagree;
//     * the paged decode kernel computes the int8 dot for every one of the 64
//       page tokens and only then overwrites the score with -INFINITY when
//       tok >= n_valid or k_abs >= q_end, so poisoned K bytes / poisoned
//       k_scale in the masked tail are read and discarded;
//     * decode_v_int4_vec_strided() zero-fills V rows with tok >= n_valid
//       before the PV matmul ("0*NaN would poison PV");
//     * probabilities are stored as fp16 (sycl::half), V is dequantized into
//       fp16, the PV accumulator is fp32;
//     * k_scale / v_scale / v_zero are fp32 arrays, 4 B each.
//
// ASSUMED here, and flagged in the output:
//   hkv = 1. The shipped PageLayout refuses hkv == 1 (only 2 or 4), so a TP4
//   rank -- two KV heads, each replicated on two ranks, with 6 Q heads/rank -- cannot be
//   described by it. This fixture applies the documented per-head formulas
//   with hkv = 1 (396 B per token per head, page 25,344 B) as an
//   extrapolation, and separately asserts that the documented layout refuses
//   that geometry.
//
// Every numeric tolerance below is a DERIVED bound printed next to the
// MEASURED value it is compared against.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int D = 256;         // head_dim, documented
constexpr int GQA = 6;         // Q heads per KV head, documented
constexpr int PAGE = 64;       // kernel page, documented
constexpr int V4_COLS = D / 2; // packed int4 columns per head, documented
constexpr int SCALE_BYTES = 4; // fp32 metadata, documented
constexpr int HKV = 1;         // ASSUMED slice: one KV head per rank
constexpr int HQ = GQA * HKV;  // 6 Q heads sharing it
constexpr int DECODE_MAX_T = PAGE / GQA; // 10, documented
constexpr long SLOT_BYTES_PER_HEAD = D + V4_COLS + 3 * SCALE_BYTES; // 396

using Row = std::array<double, D>;
using QToken = std::array<Row, HQ>;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

void measured(const char* what, double value) {
    std::printf("MEASURED %s = %.12g\n", what, value);
}

void derived(const char* what, double value) {
    std::printf("DERIVED  %s = %.12g\n", what, value);
}

// ---------------------------------------------------------------- arithmetic

// Documented rounding: half away from zero (oracle.py `_rint`).
int round_half_away(double v) {
    const double m = std::floor(std::fabs(v) + 0.5);
    return v < 0.0 ? -static_cast<int>(m) : static_cast<int>(m);
}

// Rounding actually used by the SYCL store: sycl::rint, ties to even.
int round_ties_even(double v) {
    double f = std::floor(v);
    const double frac = v - f;
    if (frac > 0.5) {
        f += 1.0;
    } else if (frac == 0.5) {
        const long long fi = static_cast<long long>(f);
        if (fi % 2 != 0) f += 1.0;
    }
    return static_cast<int>(f);
}

// fp32 -> bf16 with round-to-nearest-even, returned as a float.
float to_bf16(float x) {
    std::uint32_t b = 0;
    std::memcpy(&b, &x, sizeof b);
    if ((b & 0x7F800000u) == 0x7F800000u) return x; // inf / nan untouched
    b += 0x7FFFu + ((b >> 16) & 1u);
    b &= 0xFFFF0000u;
    float out = 0.0f;
    std::memcpy(&out, &b, sizeof out);
    return out;
}

// fp32 -> fp16 with round-to-nearest-even, returned as a float (the kernel
// stores probabilities and dequantized V as sycl::half).
float to_fp16(float x) {
    if (!std::isfinite(x) || x == 0.0f) return x;
    const float a = std::fabs(x);
    if (a >= 65520.0f) return std::copysign(std::numeric_limits<float>::infinity(), x);
    double v = static_cast<double>(x);
    int e2 = 0;
    (void)std::frexp(v, &e2);
    // fp16 has an 11-bit significand; the subnormal step is 2^-24.
    const int ulp_exp = std::max(e2 - 11, -24);
    const double step = std::ldexp(1.0, ulp_exp);
    v = std::nearbyint(v / step) * step;
    if (std::fabs(v) > 65504.0) v = std::copysign(65504.0, v);
    return static_cast<float>(v);
}

float load_f32(const std::uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof v);
    return v;
}

void store_f32(std::uint8_t* p, float v) { std::memcpy(p, &v, sizeof v); }

// Deterministic rows, no dependency on the Python RNG.
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // uniform in [-1, 1]
    double uniform() {
        const std::uint64_t raw = next() >> 11;
        return (static_cast<double>(raw) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
    }
};

Row make_row(Rng& rng) {
    Row row{};
    for (int d = 0; d < D; ++d) row[static_cast<std::size_t>(d)] = rng.uniform();
    return row;
}

// ---------------------------------------------------------------- geometry

struct Geometry {
    int hkv;
    int hq;
    long k_bytes;
    long v_bytes;
    long scale_bytes;
    long page_bytes;
    long bytes_per_token;
    long fp8_bytes_per_token;
};

Geometry geometry(int hkv) {
    Geometry g{};
    g.hkv = hkv;
    g.hq = GQA * hkv;
    g.k_bytes = PAGE * hkv * D;
    g.v_bytes = PAGE * hkv * V4_COLS;
    g.scale_bytes = PAGE * hkv * SCALE_BYTES;
    g.page_bytes = g.k_bytes + g.v_bytes + 3 * g.scale_bytes;
    g.bytes_per_token = g.page_bytes / PAGE;
    g.fp8_bytes_per_token = hkv * 2 * D;
    return g;
}

// Reimplementation of layout.py PageLayout.__post_init__ / for_heads refusals.
// Empty string means accepted.
std::string geometry_refusal(int q_heads, int kv_heads) {
    if (kv_heads != 2 && kv_heads != 4)
        return "unsupported per-GPU KV head count " + std::to_string(kv_heads) +
               " (K8/V4 serves 2 or 4)";
    if (q_heads != GQA * kv_heads)
        return "K8/V4 serves 12/2 (TP2) or 24/4 (single GPU) local heads, got " +
               std::to_string(q_heads) + " Q / " + std::to_string(kv_heads) + " KV";
    return std::string();
}

// ---------------------------------------------------------------- regions

struct RegionSpec {
    const char* name;
    long byte_offset;
    long len_bytes;
    long elem_bytes;
    std::array<long, 4> shape;  // pages, tokens, heads, cells
    std::array<long, 4> stride; // in elements
};

std::array<RegionSpec, 5> region_specs(long num_pages, int hkv) {
    const Geometry g = geometry(hkv);
    const long scale0 = g.k_bytes + g.v_bytes;
    // layout.py: float_stride0 = page_bytes // SCALE_BYTES
    const long float_stride0 = g.page_bytes / SCALE_BYTES;
    return {{
        {"k", 0, g.k_bytes, 1, {num_pages, PAGE, hkv, D}, {g.page_bytes, hkv * D, D, 1}},
        {"v", g.k_bytes, g.v_bytes, 1, {num_pages, PAGE, hkv, V4_COLS}, {g.page_bytes, hkv * V4_COLS, V4_COLS, 1}},
        {"k_scale", scale0, g.scale_bytes, 4, {num_pages, PAGE, hkv}, {float_stride0, hkv, 1, 0}},
        {"v_scale", scale0 + g.scale_bytes, g.scale_bytes, 4, {num_pages, PAGE, hkv}, {float_stride0, hkv, 1, 0}},
        {"v_zero", scale0 + 2 * g.scale_bytes, g.scale_bytes, 4, {num_pages, PAGE, hkv}, {float_stride0, hkv, 1, 0}},
    }};
}

// layout.py element_byte() for a rank-3 or rank-4 region.
long element_byte(const RegionSpec& spec, int rank, long p, long t, long h, long c) {
    const std::array<long, 4> idx{p, t, h, c};
    long off = spec.byte_offset;
    for (int i = 0; i < rank; ++i)
        off += idx[static_cast<std::size_t>(i)] * spec.stride[static_cast<std::size_t>(i)] * spec.elem_bytes;
    return off;
}

// Address helpers for the hkv=1 page buffer used by the store/load paths.
long k_addr(long page_bytes, int page, int tok, int head, int d) {
    return static_cast<long>(page) * page_bytes + static_cast<long>(tok) * (HKV * D) + head * D + d;
}

long v_addr(long page_bytes, int page, int tok, int head, int cell) {
    return static_cast<long>(page) * page_bytes + PAGE * HKV * D +
           static_cast<long>(tok) * (HKV * V4_COLS) + head * V4_COLS + cell;
}

// which: 0 = k_scale, 1 = v_scale, 2 = v_zero
long meta_addr(long page_bytes, int page, int tok, int head, int which) {
    return static_cast<long>(page) * page_bytes + PAGE * HKV * D + PAGE * HKV * V4_COLS +
           static_cast<long>(which) * (PAGE * HKV * SCALE_BYTES) +
           static_cast<long>(tok) * (HKV * SCALE_BYTES) + head * SCALE_BYTES;
}

// ---------------------------------------------------------------- quantization

struct KQuant {
    std::array<std::int8_t, D> q{};
    float scale = 0.0f;
};

struct VQuant {
    std::array<std::uint8_t, V4_COLS> packed{};
    float scale = 0.0f;
    float zero = 0.0f;
};

// Documented K/Q rule. `rounder` selects the documented (half-away) or the
// source-observed (rint / ties-to-even) tie behaviour.
KQuant quant_k(const Row& row, bool ties_even) {
    double amax = 0.0;
    for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(row[static_cast<std::size_t>(d)]));
    KQuant out{};
    if (amax == 0.0) return out; // scale 0, all lanes 0
    out.scale = static_cast<float>(amax / 127.0);
    const float inv = 1.0f / out.scale;
    for (int d = 0; d < D; ++d) {
        const double scaled = row[static_cast<std::size_t>(d)] * static_cast<double>(inv);
        int q = round_half_away(scaled);
        if (ties_even) q = round_ties_even(scaled);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        out.q[static_cast<std::size_t>(d)] = static_cast<std::int8_t>(q);
    }
    return out;
}

// Injected-scale entry point: exercises the documented +-127 clamp, which the
// amax/127 rule cannot reach on its own.
KQuant quant_k_with_scale(const Row& row, float scale) {
    KQuant out{};
    out.scale = scale;
    const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int d = 0; d < D; ++d) {
        int q = round_half_away(row[static_cast<std::size_t>(d)] * static_cast<double>(inv));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        out.q[static_cast<std::size_t>(d)] = static_cast<std::int8_t>(q);
    }
    return out;
}

std::array<double, D> dequant_k(const KQuant& k) {
    std::array<double, D> out{};
    for (int d = 0; d < D; ++d)
        out[static_cast<std::size_t>(d)] = static_cast<double>(k.q[static_cast<std::size_t>(d)]) *
                                           static_cast<double>(k.scale);
    return out;
}

// Documented V affine rule, unchanged from oracle.py and the SYCL V store.
VQuant quant_v(const Row& row) {
    double lo = row[0], hi = row[0];
    for (int d = 1; d < D; ++d) {
        lo = std::min(lo, row[static_cast<std::size_t>(d)]);
        hi = std::max(hi, row[static_cast<std::size_t>(d)]);
    }
    VQuant out{};
    out.scale = static_cast<float>(std::max((hi - lo) / 15.0, 1.0e-8));
    double zero = std::floor(-lo / static_cast<double>(out.scale) + 0.5);
    zero = std::min(15.0, std::max(0.0, zero));
    out.zero = static_cast<float>(zero);
    for (int b = 0; b < V4_COLS; ++b) {
        std::uint8_t byte = 0;
        for (int half = 0; half < 2; ++half) {
            const int d = 2 * b + half; // even dimension is the low nibble
            double q = std::floor(row[static_cast<std::size_t>(d)] / static_cast<double>(out.scale) +
                                  zero + 0.5);
            q = std::min(15.0, std::max(0.0, q));
            const std::uint8_t nib = static_cast<std::uint8_t>(q);
            byte = static_cast<std::uint8_t>(byte | (half == 0 ? nib : static_cast<std::uint8_t>(nib << 4)));
        }
        out.packed[static_cast<std::size_t>(b)] = byte;
    }
    return out;
}

std::array<double, D> dequant_v(const VQuant& v) {
    std::array<double, D> out{};
    for (int b = 0; b < V4_COLS; ++b) {
        const std::uint8_t byte = v.packed[static_cast<std::size_t>(b)];
        for (int half = 0; half < 2; ++half) {
            const int q = half == 0 ? (byte & 0x0F) : (byte >> 4);
            out[static_cast<std::size_t>(2 * b + half)] =
                (static_cast<double>(q) - static_cast<double>(v.zero)) * static_cast<double>(v.scale);
        }
    }
    return out;
}

// ---------------------------------------------------------------- causal lens

struct Lens {
    int max_end;
    std::vector<int> ends;
    bool refused;
    std::string reason;
};

// layout.py visible_lens(): index 0 is the longest end, 1+j the end of query j,
// each clamped at 1. q_len outside the packed decode tile is refused.
Lens visible_lens(int seq_len, int q_len) {
    Lens lens{};
    if (q_len < 1 || q_len > DECODE_MAX_T) {
        lens.refused = true;
        lens.reason = "q_len outside packed decode tile";
        lens.max_end = 0;
        return lens;
    }
    for (int j = 0; j < q_len; ++j) {
        const int end = seq_len - (q_len - 1 - j);
        lens.ends.push_back(end >= 1 ? end : 1);
    }
    lens.max_end = *std::max_element(lens.ends.begin(), lens.ends.end());
    return lens;
}

// layout.py builtin_q_end(): the non-varlen kernel end.
int builtin_q_end(int seq_len, int q_len, int q_tok) { return seq_len - q_len + q_tok + 1; }

// ---------------------------------------------------------------- cache rows

struct CacheRow {
    KQuant k;
    VQuant v;
    Row vdeq{};
};

std::vector<CacheRow> build_cache(const std::vector<Row>& keys, const std::vector<Row>& values) {
    std::vector<CacheRow> cache(keys.size());
    for (std::size_t t = 0; t < keys.size(); ++t) {
        cache[t].k = quant_k(keys[t], false);
        cache[t].v = quant_v(values[t]);
        cache[t].vdeq = dequant_v(cache[t].v);
    }
    return cache;
}

// ---------------------------------------------------------------- attention

const double INV_SQRT_D = 1.0 / std::sqrt(static_cast<double>(D));

// Score row for one query head, documented formula, tokens >= end masked.
std::vector<double> scores_fp64(const Row& qrow, const std::vector<CacheRow>& cache, int seq_len,
                                int end) {
    const KQuant qq = quant_k(qrow, false);
    std::vector<double> s(static_cast<std::size_t>(seq_len), -std::numeric_limits<double>::infinity());
    for (int t = 0; t < seq_len; ++t) {
        if (t >= end) continue;
        int acc = 0;
        for (int d = 0; d < D; ++d)
            acc += static_cast<int>(qq.q[static_cast<std::size_t>(d)]) *
                   static_cast<int>(cache[static_cast<std::size_t>(t)].k.q[static_cast<std::size_t>(d)]);
        s[static_cast<std::size_t>(t)] = static_cast<double>(acc) * static_cast<double>(qq.scale) *
                                         static_cast<double>(cache[static_cast<std::size_t>(t)].k.scale) *
                                         INV_SQRT_D;
    }
    return s;
}

// Independently wired comparison path: same quantization, but the score is
// rebuilt from bf16-rounded dequantized operands and probabilities / output are
// bf16, i.e. what a stock BF16 GQA attention would carry.
std::vector<double> scores_bf16(const Row& qrow, const std::vector<CacheRow>& cache, int seq_len,
                                int end) {
    const KQuant qq = quant_k(qrow, false);
    std::array<float, D> qf{};
    for (int d = 0; d < D; ++d)
        qf[static_cast<std::size_t>(d)] =
            to_bf16(static_cast<float>(static_cast<double>(qq.q[static_cast<std::size_t>(d)]) *
                                       static_cast<double>(qq.scale)));
    std::vector<double> s(static_cast<std::size_t>(seq_len), -std::numeric_limits<double>::infinity());
    for (int t = 0; t < seq_len; ++t) {
        if (t >= end) continue;
        double acc = 0.0;
        for (int d = 0; d < D; ++d) {
            const float kf = to_bf16(static_cast<float>(
                static_cast<double>(cache[static_cast<std::size_t>(t)].k.q[static_cast<std::size_t>(d)]) *
                static_cast<double>(cache[static_cast<std::size_t>(t)].k.scale)));
            acc += static_cast<double>(qf[static_cast<std::size_t>(d)]) * static_cast<double>(kf);
        }
        s[static_cast<std::size_t>(t)] = acc * INV_SQRT_D;
    }
    return s;
}

// Stable softmax (max subtraction) over a score vector, masked entries are -inf.
std::vector<double> softmax_stable(const std::vector<double>& s) {
    std::vector<double> p(s.size(), 0.0);
    double m = -std::numeric_limits<double>::infinity();
    for (double v : s) m = std::max(m, v);
    if (m == -std::numeric_limits<double>::infinity()) return p;
    double sum = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        p[i] = std::exp(s[i] - m);
        sum += p[i];
    }
    if (sum == 0.0) return std::vector<double>(s.size(), 0.0);
    const double inv = 1.0 / sum;
    for (double& v : p) v *= inv;
    return p;
}

// Unshifted softmax, only used to show that the stable form is load bearing.
std::vector<double> softmax_naive(const std::vector<double>& s) {
    std::vector<double> p(s.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        p[i] = std::exp(s[i]);
        sum += p[i];
    }
    if (!std::isfinite(sum) || sum == 0.0) return std::vector<double>(s.size(),
                                                                     std::numeric_limits<double>::quiet_NaN());
    const double inv = 1.0 / sum;
    for (double& v : p) v *= inv;
    return p;
}

// Reference A: fp64, masked tokens skipped, exact double dequant of V.
std::vector<QToken> attention_reference(const std::vector<QToken>& q,
                                        const std::vector<CacheRow>& cache, int seq_len, int q_len) {
    std::vector<QToken> out(static_cast<std::size_t>(q_len));
    const Lens lens = visible_lens(seq_len, q_len);
    for (int j = 0; j < q_len; ++j) {
        const int end = lens.ends[static_cast<std::size_t>(j)];
        for (int h = 0; h < HQ; ++h) {
            const std::vector<double> s = scores_fp64(q[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)],
                                                      cache, seq_len, end);
            const std::vector<double> p = softmax_stable(s);
            Row acc{};
            for (int t = 0; t < seq_len; ++t) {
                const double pr = p[static_cast<std::size_t>(t)];
                if (pr == 0.0) continue;
                for (int d = 0; d < D; ++d)
                    acc[static_cast<std::size_t>(d)] +=
                        pr * cache[static_cast<std::size_t>(t)].vdeq[static_cast<std::size_t>(d)];
            }
            out[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)] = acc;
        }
    }
    return out;
}

// Reference B: bf16-rounded operands / probabilities, bf16 output, fp32
// accumulation. Written independently of A on purpose.
std::vector<QToken> attention_bf16(const std::vector<QToken>& q, const std::vector<CacheRow>& cache,
                                   int seq_len, int q_len) {
    std::vector<QToken> out(static_cast<std::size_t>(q_len));
    const Lens lens = visible_lens(seq_len, q_len);
    for (int j = 0; j < q_len; ++j) {
        const int end = lens.ends[static_cast<std::size_t>(j)];
        for (int h = 0; h < HQ; ++h) {
            const std::vector<double> s = scores_bf16(q[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)],
                                                      cache, seq_len, end);
            std::vector<double> p = softmax_stable(s);
            for (double& v : p) v = static_cast<double>(to_bf16(static_cast<float>(v)));
            std::array<float, D> acc{};
            for (int t = 0; t < seq_len; ++t) {
                const float pr = static_cast<float>(p[static_cast<std::size_t>(t)]);
                if (pr == 0.0f) continue;
                for (int d = 0; d < D; ++d) {
                    const float vb = to_bf16(static_cast<float>(
                        cache[static_cast<std::size_t>(t)].vdeq[static_cast<std::size_t>(d)]));
                    acc[static_cast<std::size_t>(d)] += pr * vb;
                }
            }
            Row row{};
            for (int d = 0; d < D; ++d)
                row[static_cast<std::size_t>(d)] = static_cast<double>(
                    to_bf16(acc[static_cast<std::size_t>(d)]));
            out[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)] = row;
        }
    }
    return out;
}

// Store one token of one KV head into a page buffer using the documented
// region order (hkv = 1).
void store_token(std::vector<std::uint8_t>& pages, long page_bytes, int tok, const KQuant& k,
                 const VQuant& v) {
    const int page = tok / PAGE;
    const int t = tok % PAGE;
    for (int d = 0; d < D; ++d)
        pages[static_cast<std::size_t>(k_addr(page_bytes, page, t, 0, d))] =
            static_cast<std::uint8_t>(k.q[static_cast<std::size_t>(d)]);
    for (int b = 0; b < V4_COLS; ++b)
        pages[static_cast<std::size_t>(v_addr(page_bytes, page, t, 0, b))] = v.packed[static_cast<std::size_t>(b)];
    store_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, page, t, 0, 0))], k.scale);
    store_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, page, t, 0, 1))], v.scale);
    store_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, page, t, 0, 2))], v.zero);
}

// Kernel-shaped path: per page, the int8 dot is computed for all 64 tokens and
// only then masked; V rows with tok >= n_valid are treated as zero before PV;
// page-normalized probabilities pass through fp16 and the page partials are
// merged with the documented M/L rescaling in fp32 (the S1 + S2 split path;
// a single page degenerates to plain S1). Poisoned bytes in the masked tail are
// read here and must not reach the result.
std::vector<QToken> attention_kernel_shaped(const std::vector<QToken>& q,
                                            const std::vector<CacheRow>& cache,
                                            const std::vector<std::uint8_t>& pages, long page_bytes,
                                            int seq_len, int q_len) {
    (void)cache; // the page buffer is the only KV source in this path
    std::vector<QToken> out(static_cast<std::size_t>(q_len));
    const Lens lens = visible_lens(seq_len, q_len);
    const int n_pages = (seq_len + PAGE - 1) / PAGE;
    for (int j = 0; j < q_len; ++j) {
        const int end = lens.ends[static_cast<std::size_t>(j)];
        for (int h = 0; h < HQ; ++h) {
            const KQuant qq = quant_k(q[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)], false);
            std::array<float, D> acc{};
            float run_m = -std::numeric_limits<float>::infinity();
            float run_l = 0.0f;
            for (int pg = 0; pg < n_pages; ++pg) {
                const int n_valid = std::max(0, std::min(PAGE, seq_len - pg * PAGE));
                if (n_valid <= 0) continue;
                std::array<float, PAGE> s{};
                for (int tok = 0; tok < PAGE; ++tok) {
                    const float ks = load_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, pg, tok, 0, 0))]);
                    int acc8 = 0;
                    for (int d = 0; d < D; ++d) {
                        const std::int8_t kv = static_cast<std::int8_t>(
                            pages[static_cast<std::size_t>(k_addr(page_bytes, pg, tok, 0, d))]);
                        acc8 += static_cast<int>(qq.q[static_cast<std::size_t>(d)]) * static_cast<int>(kv);
                    }
                    float score = static_cast<float>(acc8) * qq.scale * ks *
                                  static_cast<float>(INV_SQRT_D);
                    const long long k_abs = static_cast<long long>(pg) * PAGE + tok;
                    if (tok >= n_valid || k_abs >= end) score = -std::numeric_limits<float>::infinity();
                    s[static_cast<std::size_t>(tok)] = score;
                }
                float m = -std::numeric_limits<float>::infinity();
                for (float v : s) m = std::max(m, v);
                std::array<float, PAGE> p16{};
                float sum = 0.0f;
                if (m != -std::numeric_limits<float>::infinity()) {
                    std::array<float, PAGE> e{};
                    for (int tok = 0; tok < PAGE; ++tok) {
                        e[static_cast<std::size_t>(tok)] = std::exp(s[static_cast<std::size_t>(tok)] - m);
                        sum += e[static_cast<std::size_t>(tok)];
                    }
                    const float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
                    for (int tok = 0; tok < PAGE; ++tok)
                        p16[static_cast<std::size_t>(tok)] = to_fp16(e[static_cast<std::size_t>(tok)] * inv);
                }
                // Per-page PV partial over fp16 probabilities, fp32 accumulation.
                std::array<float, D> page_acc{};
                for (int tok = 0; tok < PAGE; ++tok) {
                    // Guard under test: never read affine metadata or nibbles
                    // for a row outside the page's valid length.
                    if (tok >= n_valid) continue;
                    const float pr = p16[static_cast<std::size_t>(tok)];
                    if (pr == 0.0f) continue;
                    const float vs = load_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, pg, tok, 0, 1))]);
                    const float vz = load_f32(&pages[static_cast<std::size_t>(meta_addr(page_bytes, pg, tok, 0, 2))]);
                    for (int b = 0; b < V4_COLS; ++b) {
                        const std::uint8_t byte = pages[static_cast<std::size_t>(v_addr(page_bytes, pg, tok, 0, b))];
                        for (int half = 0; half < 2; ++half) {
                            const int nib = half == 0 ? (byte & 0x0F) : (byte >> 4);
                            const float vb = to_fp16((static_cast<float>(nib) - vz) * vs);
                            const int d = 2 * b + half;
                            page_acc[static_cast<std::size_t>(d)] += pr * vb;
                        }
                    }
                }
                // S2 merge: rescale the running accumulator to the new maximum.
                if (m == -std::numeric_limits<float>::infinity()) continue;
                const float new_m = std::max(run_m, m);
                const float old_scale =
                    run_m == -std::numeric_limits<float>::infinity() ? 0.0f : std::exp(run_m - new_m);
                const float page_weight = std::exp(m - new_m) * sum;
                for (int d = 0; d < D; ++d)
                    acc[static_cast<std::size_t>(d)] = acc[static_cast<std::size_t>(d)] * old_scale +
                                                       page_acc[static_cast<std::size_t>(d)] * page_weight;
                run_l = run_l * old_scale + page_weight;
                run_m = new_m;
            }
            Row row{};
            const float norm = run_l == 0.0f ? 0.0f : 1.0f / run_l;
            for (int d = 0; d < D; ++d)
                row[static_cast<std::size_t>(d)] = static_cast<double>(acc[static_cast<std::size_t>(d)] * norm);
            out[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)] = row;
        }
    }
    return out;
}

double max_abs_diff(const std::vector<QToken>& a, const std::vector<QToken>& b) {
    double worst = 0.0;
    for (std::size_t j = 0; j < a.size() && j < b.size(); ++j)
        for (int h = 0; h < HQ; ++h)
            for (int d = 0; d < D; ++d)
                worst = std::max(worst, std::fabs(a[j][static_cast<std::size_t>(h)][static_cast<std::size_t>(d)] -
                                                  b[j][static_cast<std::size_t>(h)][static_cast<std::size_t>(d)]));
    return worst;
}

bool all_finite(const std::vector<QToken>& m) {
    for (const QToken& tok : m)
        for (const Row& row : tok)
            for (double v : row)
                if (!std::isfinite(v)) return false;
    return true;
}

// ---------------------------------------------------------------- tests

void test_geometry() {
    std::printf("\n== geometry: D=256, GQA=6, one KV head, plus the documented pair ==\n");
    const Geometry one = geometry(HKV);
    derived("assumed hkv=1 page bytes", static_cast<double>(one.page_bytes));
    derived("assumed hkv=1 bytes/token", static_cast<double>(one.bytes_per_token));
    derived("documented SLOT_BYTES_PER_HEAD", static_cast<double>(SLOT_BYTES_PER_HEAD));
    check(one.hq == 6, "one KV head carries exactly 6 query heads");
    check(one.bytes_per_token == SLOT_BYTES_PER_HEAD, "hkv=1 bytes/token equals the 396 B slot");
    check(one.page_bytes == 64 * 396, "hkv=1 page is 64 slots = 25344 B");
    check(one.fp8_bytes_per_token == 512, "fp8 K+V per token per head is 2*D = 512 B");

    const Geometry tp2 = geometry(2);
    const Geometry tp1 = geometry(4);
    check(tp2.page_bytes == 50688 && tp2.bytes_per_token == 792, "TP2 (12 Q / 2 KV) page 50688, 792 B/token");
    check(tp1.page_bytes == 101376 && tp1.bytes_per_token == 1584, "TP1-style (24 Q / 4 KV) page 101376, 1584 B/token");
    check(tp2.fp8_bytes_per_token == 1024 && tp1.fp8_bytes_per_token == 2048, "fp8 per token scales with hkv");

    check(!geometry_refusal(12, 2).empty() == false, "documented geometry accepts 12 Q / 2 KV");
    check(!geometry_refusal(24, 4).empty() == false, "documented geometry accepts 24 Q / 4 KV");
    const std::string r1 = geometry_refusal(6, 1);
    const std::string r2 = geometry_refusal(6, 2);
    const std::string r3 = geometry_refusal(12, 1);
    const std::string r4 = geometry_refusal(12, 3);
    std::printf("REFUSED  for_heads(6,1): %s\n", r1.c_str());
    std::printf("REFUSED  for_heads(6,2): %s\n", r2.c_str());
    std::printf("REFUSED  for_heads(12,1): %s\n", r3.c_str());
    std::printf("REFUSED  for_heads(12,3): %s\n", r4.c_str());
    check(!r1.empty(), "documented layout REFUSES the hkv=1 TP4 slice (assumed geometry here)");
    check(!r2.empty(), "documented layout refuses 6 Q heads over 2 KV heads (GQA mismatch)");
    check(!r3.empty(), "documented layout refuses 12 Q heads over 1 KV head");
    check(!r4.empty(), "documented layout refuses hkv=3");
    check(geometry_refusal(8, 2) == "K8/V4 serves 12/2 (TP2) or 24/4 (single GPU) local heads, got 8 Q / 2 KV",
          "refusal names the offending head pair");
}

void test_regions() {
    std::printf("\n== page regions, strides and fp32 metadata alignment ==\n");
    const long page_bytes = geometry(HKV).page_bytes;
    const auto specs = region_specs(1, HKV);
    const char* names[5] = {"k", "v", "k_scale", "v_scale", "v_zero"};
    for (int i = 0; i < 5; ++i) {
        std::printf("REGION   %-8s offset=%-6ld len=%-6ld elem=%ld\n", names[i], specs[static_cast<std::size_t>(i)].byte_offset,
                    specs[static_cast<std::size_t>(i)].len_bytes,
                    specs[static_cast<std::size_t>(i)].elem_bytes);
    }
    long tiled = 0;
    bool aligned = true;
    for (const RegionSpec& spec : specs) {
        tiled += spec.len_bytes;
        if (spec.byte_offset % 4 != 0) aligned = false;
    }
    measured("region bytes summed over the page", static_cast<double>(tiled));
    check(tiled == page_bytes, "the five regions tile the page exactly, no gap or overlap");
    check(aligned, "every region starts 4-byte aligned");

    // Within the page the regions are region-major: K for token 1 follows the
    // whole 64-token K block, not token 0's slot.
    check(element_byte(specs[0], 4, 0, 1, 0, 0) == D * HKV, "K token stride is hkv*D = 256 B");
    check(element_byte(specs[0], 4, 0, 64, 0, 0) == 64 * D * HKV, "K region end offset == 16384");
    check(element_byte(specs[1], 4, 0, 0, 0, 0) == 64 * D * HKV, "V starts right after the K region");
    check(element_byte(specs[2], 3, 0, 0, 0, 0) == 24576, "k_scale starts at 24576");
    check(element_byte(specs[3], 3, 0, 0, 0, 0) == 24832, "v_scale starts at 24832");
    check(element_byte(specs[4], 3, 0, 0, 0, 0) == 25088, "v_zero starts at 25088");
    check(element_byte(specs[4], 3, 0, 63, 0, 0) + SCALE_BYTES == page_bytes,
          "the last v_zero closes the page at 25344");

    // vLLM's logical view is [block, head, token, 396]; the memory order is
    // [block, token, head, cell]. Those disagree once hkv > 1.
    const auto tp2 = region_specs(2, 2);
    const long mem_step = element_byte(tp2[0], 4, 0, 1, 0, 0) - element_byte(tp2[0], 4, 0, 0, 0, 0);
    measured("TP2 memory-order token stride for K (B)", static_cast<double>(mem_step));
    derived("logical [head][token][396] token stride (B)", 396.0);
    check(mem_step == 512, "memory order advances a token by hkv*D = 512 B, not 396 B");
    check(element_byte(tp2[0], 4, 1, 0, 0, 0) == 50688, "K page stride equals the whole page");
    check(page_bytes % SCALE_BYTES == 0, "page bytes divide by 4 so the fp32 scale stride is integral");
}

void test_k_quant() {
    std::printf("\n== K/Q symmetric int8: scale, rounding ties, clamp, zero row ==\n");
    Rng rng(20261119u);
    Row row = make_row(rng);
    row[17] = 3.75; // force a known amax
    KQuant k = quant_k(row, false);
    double amax = 0.0;
    for (double v : row) amax = std::max(amax, std::fabs(v));
    measured("K amax", amax);
    measured("K scale (fp32)", static_cast<double>(k.scale));
    derived("K scale bound amax/127", amax / 127.0);
    check(std::fabs(static_cast<double>(k.scale) - amax / 127.0) < 1e-9, "K scale is amax/127");

    int qmax = 0;
    double worst = 0.0;
    const std::array<double, D> back = dequant_k(k);
    for (int d = 0; d < D; ++d) {
        qmax = std::max(qmax, std::abs(static_cast<int>(k.q[static_cast<std::size_t>(d)])));
        worst = std::max(worst, std::fabs(back[static_cast<std::size_t>(d)] - row[static_cast<std::size_t>(d)]));
    }
    measured("K |q| max", static_cast<double>(qmax));
    measured("K roundtrip max abs error", worst);
    check(qmax == 127, "the amax lane lands on +127");
    check(worst < static_cast<double>(k.scale) + 1e-6, "K roundtrip error is below one scale step");

    // The documented clamp cannot be reached from amax/127; only an injected
    // (under-estimated) scale exercises it.
    Row clip{};
    clip[0] = 127.5;
    clip[1] = -127.5;
    const KQuant injected = quant_k_with_scale(clip, 1.0f);
    measured("injected-scale q for 127.5", static_cast<double>(injected.q[0]));
    measured("injected-scale q for -127.5", static_cast<double>(injected.q[1]));
    check(injected.q[0] == 127 && injected.q[1] == -127, "clamp pins an injected overshoot to +-127");
    bool clamp_unreachable = true;
    for (int trial = 0; trial < 64; ++trial) {
        Row r = make_row(rng);
        const KQuant q = quant_k(r, false);
        for (std::int8_t v : q.q)
            if (v == -128) clamp_unreachable = false;
    }
    check(clamp_unreachable, "amax/127 alone never produces -128, so the clamp is a guard only");

    // Zero row: scale 0, zero codes, no NaN. The kernel's inv becomes 0 too.
    Row zeros{};
    const KQuant zk = quant_k(zeros, false);
    double zsum = 0.0;
    for (std::int8_t v : zk.q) zsum += std::abs(static_cast<int>(v));
    measured("zero-row K scale", static_cast<double>(zk.scale));
    check(zk.scale == 0.0f, "an all-zero K row gets scale 0");
    check(zsum == 0.0, "an all-zero K row keeps every code at 0");
    check(!std::isnan(static_cast<double>(zk.scale)) && zsum == 0.0, "zero row does not produce NaN");

    // Constructed tie row: amax = 127 gives scale exactly 1.0, so half steps are
    // exact and the two rounding rules disagree.
    Row ties{};
    ties[0] = 127.0;
    ties[1] = 0.5;
    ties[2] = -0.5;
    ties[3] = 2.5;
    ties[4] = 1.5;
    const KQuant doc = quant_k(ties, false);
    const KQuant src = quant_k(ties, true);
    measured("tie-row scale", static_cast<double>(doc.scale));
    check(doc.scale == 1.0f, "amax=127 makes the K scale exactly 1.0");
    const int doc_expect[5] = {127, 1, -1, 3, 2};
    const int src_expect[5] = {127, 0, 0, 2, 2};
    bool doc_ok = true, src_ok = true;
    int diverged = 0;
    for (int i = 0; i < 5; ++i) {
        if (static_cast<int>(doc.q[static_cast<std::size_t>(i)]) != doc_expect[i]) doc_ok = false;
        if (static_cast<int>(src.q[static_cast<std::size_t>(i)]) != src_expect[i]) src_ok = false;
        if (doc.q[static_cast<std::size_t>(i)] != src.q[static_cast<std::size_t>(i)]) ++diverged;
    }
    std::printf("ROUNDING documented(half-away) q = [%d %d %d %d %d]\n", doc_expect[0], doc_expect[1], doc_expect[2],
                doc_expect[3], doc_expect[4]);
    std::printf("ROUNDING source(rint/ties-even) q = [%d %d %d %d %d]\n", src_expect[0], src_expect[1], src_expect[2],
                src_expect[3], src_expect[4]);
    measured("tie lanes where the two rules differ", static_cast<double>(diverged));
    check(doc_ok, "documented round-half-away gives [127 1 -1 3 2]");
    check(src_ok, "source-observed rint gives [127 0 0 2 2]");
    check(diverged == 3, "the two rounding rules differ on 3 of 5 tie lanes");
}

void test_v_quant() {
    std::printf("\n== V affine int4: known packing, zero/floor, saturation ==\n");
    Row ladder{};
    for (int d = 0; d < D; ++d) ladder[static_cast<std::size_t>(d)] = static_cast<double>(d % 16);
    const VQuant vq = quant_v(ladder);
    measured("ladder V scale", static_cast<double>(vq.scale));
    measured("ladder V zero", static_cast<double>(vq.zero));
    check(vq.scale == 1.0f && vq.zero == 0.0f, "lo=0 hi=15 gives scale 1 and zero 0");
    bool packing_ok = true;
    for (int b = 0; b < V4_COLS; ++b) {
        const int even = (2 * b) % 16;
        const int odd = (2 * b + 1) % 16;
        const std::uint8_t expect = static_cast<std::uint8_t>(even | (odd << 4));
        if (vq.packed[static_cast<std::size_t>(b)] != expect) packing_ok = false;
    }
    std::printf("PACKING  byte[0]=0x%02x byte[1]=0x%02x byte[127]=0x%02x\n", vq.packed[0], vq.packed[1], vq.packed[127]);
    check(packing_ok, "byte = even_dim | odd_dim << 4 across all 128 packed bytes");
    check(vq.packed[0] == 0x10, "byte 0 packs dims (0,1) as 0x10: low nibble is the even dimension");
    const std::array<double, D> vback = dequant_v(vq);
    double worst = 0.0;
    for (int d = 0; d < D; ++d) worst = std::max(worst, std::fabs(vback[static_cast<std::size_t>(d)] - ladder[static_cast<std::size_t>(d)]));
    measured("ladder V roundtrip max abs error", worst);
    check(worst == 0.0, "integer ladder is reproduced exactly");

    // Zero row.
    Row zrow{};
    const VQuant zv = quant_v(zrow);
    measured("zero-row V scale", static_cast<double>(zv.scale));
    measured("zero-row V zero", static_cast<double>(zv.zero));
    check(zv.scale == static_cast<float>(1e-8), "an all-zero V row falls to the 1e-8 scale floor");
    check(zv.zero == 0.0f, "zero row keeps zero-point 0");
    const std::array<double, D> zdeq = dequant_v(zv);
    double zerr = 0.0;
    for (double v : zdeq) zerr = std::max(zerr, std::fabs(v));
    measured("zero-row V dequant max magnitude", zerr);
    check(zerr == 0.0, "all-zero V row dequantizes to exactly zero");

    // Scale floor with a constant positive row: the documented edge case.
    Row cpos{};
    for (int d = 0; d < D; ++d) cpos[static_cast<std::size_t>(d)] = 0.5;
    const VQuant cp = quant_v(cpos);
    const std::array<double, D> cpdeq = dequant_v(cp);
    measured("constant +0.5 V scale", static_cast<double>(cp.scale));
    measured("constant +0.5 V zero", static_cast<double>(cp.zero));
    measured("constant +0.5 V dequant value", cpdeq[0]);
    derived("constant +0.5 saturation value (15)*1e-8", 15.0 * 1e-8);
    check(cp.scale == static_cast<float>(1e-8), "flat row hits the scale floor");
    check(cp.zero == 0.0f, "positive flat row clamps the zero-point to 0");
    check(cp.packed[0] == 0xFF, "positive flat row saturates every nibble to 15");
    check(std::fabs(cpdeq[0] - 1.5e-7) < 1e-12, "positive flat 0.5 dequantizes to 1.5e-7, not 0.5");
    measured("constant +0.5 abs error", std::fabs(cpdeq[0] - 0.5));

    Row cneg{};
    for (int d = 0; d < D; ++d) cneg[static_cast<std::size_t>(d)] = -0.5;
    const VQuant cn = quant_v(cneg);
    const std::array<double, D> cndeq = dequant_v(cn);
    measured("constant -0.5 V zero", static_cast<double>(cn.zero));
    measured("constant -0.5 V dequant value", cndeq[0]);
    check(cn.zero == 15.0f, "negative flat row clamps the zero-point to 15");
    check(cn.packed[0] == 0x00, "negative flat row saturates every nibble to 0");
    check(std::fabs(cndeq[0] + 1.5e-7) < 1e-12, "negative flat -0.5 dequantizes to -1.5e-7");
    measured("constant -0.5 abs error", std::fabs(cndeq[0] + 0.5));

    // Documented open edge case: an all-negative spread row drives the raw
    // zero-point past 15, the clamp then collapses the whole row to one code.
    Row neg{-90.0, -100.0};
    for (int d = 2; d < D; ++d) neg[static_cast<std::size_t>(d)] = -90.0 - (d % 11);
    const VQuant nv = quant_v(neg);
    const std::array<double, D> ndeq = dequant_v(nv);
    measured("all-negative row scale", static_cast<double>(nv.scale));
    measured("all-negative row zero (clamped)", static_cast<double>(nv.zero));
    derived("all-negative row raw zero-point floor(-lo/scale+0.5)", std::floor(100.0 / (10.0 / 15.0) + 0.5));
    int codes_used = 0;
    for (int q = 0; q < 16; ++q) {
        bool used = false;
        for (int b = 0; b < V4_COLS; ++b) {
            const std::uint8_t byte = nv.packed[static_cast<std::size_t>(b)];
            if ((byte & 0x0F) == q || (byte >> 4) == q) used = true;
        }
        if (used) ++codes_used;
    }
    double nerr = 0.0;
    for (int d = 0; d < D; ++d) nerr = std::max(nerr, std::fabs(ndeq[static_cast<std::size_t>(d)] - neg[static_cast<std::size_t>(d)]));
    measured("all-negative row distinct codes used", static_cast<double>(codes_used));
    measured("all-negative row max abs error", nerr);
    std::printf("OBSERVED all-negative row dequantizes to a constant %.6g\n", ndeq[0]);
    check(nv.zero == 15.0f, "raw zero-point 150 clamps to 15 (documented clamp)");
    check(codes_used == 1, "only one of the 16 codes survives, so the row is destroyed");
    check(nerr > 50.0, "no V rounding formula is assumed here: the documented rule loses >50 absolute");

    // Normal mixed-sign row: error stays inside one scale step, as the
    // package's own layout test asserts.
    Rng rng(0xC0FFEEu);
    Row mixed = make_row(rng);
    const VQuant mv = quant_v(mixed);
    const std::array<double, D> mdeq = dequant_v(mv);
    double merr = 0.0;
    for (int d = 0; d < D; ++d) merr = std::max(merr, std::fabs(mdeq[static_cast<std::size_t>(d)] - mixed[static_cast<std::size_t>(d)]));
    measured("mixed-row V scale", static_cast<double>(mv.scale));
    measured("mixed-row V roundtrip max abs error", merr);
    check(merr < static_cast<double>(mv.scale) + 1e-5, "mixed-row V roundtrip error is below one scale step");
}

void test_metadata() {
    std::printf("\n== fp32 scale metadata ==\n");
    const long page_bytes = geometry(HKV).page_bytes;
    std::vector<std::uint8_t> page(static_cast<std::size_t>(page_bytes), 0);
    const float tiny = 1.0e-7f;
    store_f32(&page[static_cast<std::size_t>(meta_addr(page_bytes, 0, 0, 0, 0))], tiny);
    const float back = load_f32(&page[static_cast<std::size_t>(meta_addr(page_bytes, 0, 0, 0, 0))]);
    measured("fp32 k_scale readback", static_cast<double>(back));
    check(back == tiny, "a 1e-7 scale survives the page as fp32 bit for bit");
    const float as_half = to_fp16(tiny);
    measured("same 1e-7 through fp16", static_cast<double>(as_half));
    check(as_half != tiny, "fp16 metadata would have changed that scale (why 4 B is used)");
    const auto specs = region_specs(1, HKV);
    check(specs[2].elem_bytes == 4 && specs[3].elem_bytes == 4 && specs[4].elem_bytes == 4,
          "k_scale, v_scale and v_zero are 4-byte regions");
}

void test_attention() {
    std::printf("\n== causal GQA: 6 query heads, 1 KV head, D=256, seq=137, q_len=3 ==\n");
    const int seq_len = 137, q_len = 3;
    Rng rng(0xA771u);
    std::vector<Row> keys(static_cast<std::size_t>(seq_len)), values(static_cast<std::size_t>(seq_len));
    for (int t = 0; t < seq_len; ++t) {
        keys[static_cast<std::size_t>(t)] = make_row(rng);
        values[static_cast<std::size_t>(t)] = make_row(rng);
    }
    std::vector<QToken> q(static_cast<std::size_t>(q_len));
    for (int j = 0; j < q_len; ++j)
        for (int h = 0; h < HQ; ++h) q[static_cast<std::size_t>(j)][static_cast<std::size_t>(h)] = make_row(rng);

    const Lens lens = visible_lens(seq_len, q_len);
    std::printf("LENS     max_end=%d ends=[%d %d %d]\n", lens.max_end, lens.ends[0], lens.ends[1], lens.ends[2]);
    check(lens.max_end == seq_len, "the longest visible end is seq_len");
    bool builtin_ok = true;
    for (int j = 0; j < q_len; ++j)
        if (lens.ends[static_cast<std::size_t>(j)] != builtin_q_end(seq_len, q_len, j)) builtin_ok = false;
    check(builtin_ok, "visible_lens ends agree with builtin_q_end for a full-length cache");
    const Lens clamped = visible_lens(1, 3);
    check(clamped.ends[0] == 1 && clamped.ends[1] == 1 && clamped.ends[2] == 1,
          "ends clamp at 1 when the cache is shorter than the decode tile");
    check(visible_lens(seq_len, DECODE_MAX_T + 1).refused, "q_len above the 10-token packed tile is refused");

    const std::vector<CacheRow> cache = build_cache(keys, values);
    const std::vector<QToken> ref = attention_reference(q, cache, seq_len, q_len);
    const std::vector<QToken> bf = attention_bf16(q, cache, seq_len, q_len);

    std::vector<std::uint8_t> pages(static_cast<std::size_t>(3 * geometry(HKV).page_bytes), 0);
    const long page_bytes = geometry(HKV).page_bytes;
    for (int t = 0; t < seq_len; ++t) store_token(pages, page_bytes, t, cache[static_cast<std::size_t>(t)].k,
                                                 cache[static_cast<std::size_t>(t)].v);
    const std::vector<QToken> kern = attention_kernel_shaped(q, cache, pages, page_bytes, seq_len, q_len);
    check(all_finite(kern), "kernel-shaped fp16-prob path stays finite");

    const double d_kern = max_abs_diff(ref, kern);
    const double d_bf = max_abs_diff(ref, bf);
    measured("max |reference - kernel_shaped| (fp16 probs, fp16 V, fp32 accum)", d_kern);
    measured("max |reference - bf16 reference|", d_bf);
    derived("kernel_shaped bound: fp16 prob/V rel 2^-11 on |v|<=1 plus fp32 accum", 5e-3);
    derived("bf16 bound: 4 bf16 stages at rel 2^-9 on |out|<=1", 1e-2);
    check(d_kern < 5e-3, "kernel-shaped path matches the fp64 reference inside the fp16 bound");
    check(d_bf < 1e-2, "independently wired bf16 reference matches inside the bf16 bound");

    // Causal correctness: the last key is visible to query 2 only.
    std::vector<Row> keys2 = keys, values2 = values;
    keys2[static_cast<std::size_t>(seq_len - 1)] = Row{};
    values2[static_cast<std::size_t>(seq_len - 1)] = Row{};
    const std::vector<CacheRow> cache2 = build_cache(keys2, values2);
    const std::vector<QToken> cut = attention_reference(q, cache2, seq_len, q_len);
    const double same0 = max_abs_diff({ref[0]}, {cut[0]});
    const double same1 = max_abs_diff({ref[1]}, {cut[1]});
    const double moved2 = max_abs_diff({ref[2]}, {cut[2]});
    measured("query 0 shift when the last key is zeroed", same0);
    measured("query 1 shift when the last key is zeroed", same1);
    measured("query 2 shift when the last key is zeroed", moved2);
    check(same0 == 0.0, "query 0 is untouched by the token it cannot see");
    check(same1 == 0.0, "query 1 is untouched by the token it cannot see");
    check(moved2 > 0.0, "query 2 does see it, so the mask boundary is not off by one");

    // Softmax stability: shift invariance and naive overflow.
    std::vector<double> s{};
    for (int t = 0; t < 16; ++t) s.push_back(-30.0 + 4.0 * t);
    const std::vector<double> p_stable = softmax_stable(s);
    std::vector<double> shifted = s;
    for (double& v : shifted) v += 1.0e4;
    const std::vector<double> p_shifted = softmax_stable(shifted);
    double shift_diff = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) shift_diff = std::max(shift_diff, std::fabs(p_stable[i] - p_shifted[i]));
    measured("stable softmax shift-invariance error at +1e4", shift_diff);
    check(shift_diff < 1e-12, "max-subtracted softmax is invariant to a +1e4 score shift");
    const std::vector<double> p_naive = softmax_naive(shifted);
    check(std::isnan(p_naive[0]), "the unshifted softmax overflows to NaN on the same scores");

    // Single-token attention must return exactly the dequantized V row.
    {
        std::vector<Row> k1{keys[0]}, v1{values[0]};
        std::vector<QToken> q1(1);
        for (int h = 0; h < HQ; ++h) q1[0][static_cast<std::size_t>(h)] = q[0][static_cast<std::size_t>(h)];
        const std::vector<CacheRow> c1 = build_cache(k1, v1);
        const std::vector<QToken> o1 = attention_reference(q1, c1, 1, 1);
        double single = 0.0;
        for (int h = 0; h < HQ; ++h)
            for (int d = 0; d < D; ++d)
                single = std::max(single, std::fabs(o1[0][static_cast<std::size_t>(h)][static_cast<std::size_t>(d)] -
                                                    c1[0].vdeq[static_cast<std::size_t>(d)]));
        measured("single-token attention vs dequantized V", single);
        check(single < 1e-12, "a one-token cache returns the V row itself");
    }

    // Zero K and zero Q are degenerate but defined: all scores 0, uniform
    // attention over the visible range.
    {
        std::vector<Row> kz(static_cast<std::size_t>(seq_len), Row{});
        const std::vector<CacheRow> cz = build_cache(kz, values);
        std::vector<QToken> qz(1);
        for (int h = 0; h < HQ; ++h) qz[0][static_cast<std::size_t>(h)] = q[0][static_cast<std::size_t>(h)];
        const std::vector<QToken> oz = attention_reference(qz, cz, seq_len, 1);
        Row mean{};
        for (int t = 0; t < seq_len; ++t)
            for (int d = 0; d < D; ++d) mean[static_cast<std::size_t>(d)] += cz[static_cast<std::size_t>(t)].vdeq[static_cast<std::size_t>(d)];
        for (double& v : mean) v /= seq_len;
        double zdiff = 0.0;
        for (int d = 0; d < D; ++d)
            zdiff = std::max(zdiff, std::fabs(oz[0][0][static_cast<std::size_t>(d)] - mean[static_cast<std::size_t>(d)]));
        measured("zero-K uniform attention vs mean of V", zdiff);
        check(zdiff < 1e-12, "zero K yields uniform attention over the visible tokens");
        check(all_finite(oz), "zero K and zero scale do not produce NaN");
    }

    // int8 dot headroom for D=256.
    {
        KQuant a{}, b{};
        for (int d = 0; d < D; ++d) {
            a.q[static_cast<std::size_t>(d)] = 127;
            b.q[static_cast<std::size_t>(d)] = 127;
        }
        int acc = 0;
        for (int d = 0; d < D; ++d) acc += static_cast<int>(a.q[static_cast<std::size_t>(d)]) * static_cast<int>(b.q[static_cast<std::size_t>(d)]);
        measured("max int8 dot for D=256", static_cast<double>(acc));
        derived("documented int32 accumulator ceiling", 2147483647.0);
        check(acc == 256 * 127 * 127 && acc < std::numeric_limits<std::int32_t>::max(),
              "the worst-case int8 dot fits an int32 accumulator with room to spare");
    }
}

void test_poison() {
    std::printf("\n== mutable prefix/tail validity and poisoned future rows ==\n");
    const int seq_len = 137;
    const int q_len = 1;
    const long page_bytes = geometry(HKV).page_bytes;
    Rng rng(0x9015u);
    std::vector<Row> keys(static_cast<std::size_t>(seq_len)), values(static_cast<std::size_t>(seq_len));
    for (int t = 0; t < seq_len; ++t) {
        keys[static_cast<std::size_t>(t)] = make_row(rng);
        values[static_cast<std::size_t>(t)] = make_row(rng);
    }
    std::vector<QToken> q(1);
    for (int h = 0; h < HQ; ++h) q[0][static_cast<std::size_t>(h)] = make_row(rng);

    const std::vector<CacheRow> cache = build_cache(keys, values);
    std::vector<std::uint8_t> pages(static_cast<std::size_t>(3 * page_bytes), 0);
    for (int t = 0; t < seq_len; ++t)
        store_token(pages, page_bytes, t, cache[static_cast<std::size_t>(t)].k, cache[static_cast<std::size_t>(t)].v);

    // Validity of the three pages: 64, 64 and 9 tokens.
    const int n_valid[3] = {std::max(0, std::min(PAGE, seq_len - 0 * PAGE)),
                            std::max(0, std::min(PAGE, seq_len - 1 * PAGE)),
                            std::max(0, std::min(PAGE, seq_len - 2 * PAGE))};
    std::printf("VALIDITY page valid = [%d %d %d]\n", n_valid[0], n_valid[1], n_valid[2]);
    check(n_valid[0] == 64 && n_valid[1] == 64 && n_valid[2] == 9, "137 tokens give pages of 64/64/9");

    const std::vector<QToken> clean = attention_kernel_shaped(q, cache, pages, page_bytes, seq_len, q_len);
    check(all_finite(clean), "clean paged run is finite");

    // Poison the whole masked tail of page 2: K bytes, k_scale, v_scale and
    // v_zero, all of them NaN/inf/garbage.
    std::vector<std::uint8_t> poisoned = pages;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (int tok = n_valid[2]; tok < PAGE; ++tok) {
        for (int d = 0; d < D; ++d)
            poisoned[static_cast<std::size_t>(k_addr(page_bytes, 2, tok, 0, d))] = static_cast<std::uint8_t>(0x7F);
        for (int b = 0; b < V4_COLS; ++b)
            poisoned[static_cast<std::size_t>(v_addr(page_bytes, 2, tok, 0, b))] = static_cast<std::uint8_t>(0xFF);
        store_f32(&poisoned[static_cast<std::size_t>(meta_addr(page_bytes, 2, tok, 0, 0))], nan);
        store_f32(&poisoned[static_cast<std::size_t>(meta_addr(page_bytes, 2, tok, 0, 1))], inf);
        store_f32(&poisoned[static_cast<std::size_t>(meta_addr(page_bytes, 2, tok, 0, 2))], 1.0e30f);
    }
    const std::vector<QToken> after = attention_kernel_shaped(q, cache, poisoned, page_bytes, seq_len, q_len);
    const double d_poison = max_abs_diff(clean, after);
    measured("max |clean - poisoned-future-tail|", d_poison);
    check(all_finite(after), "a NaN/inf poisoned masked tail leaves the result finite");
    check(d_poison == 0.0, "a poisoned masked tail changes the output by exactly nothing");

    // Negative control: poison a live row instead. If this changed nothing the
    // previous check would prove nothing.
    std::vector<std::uint8_t> live = pages;
    for (int d = 0; d < D; ++d)
        live[static_cast<std::size_t>(k_addr(page_bytes, 0, 0, 0, d))] = static_cast<std::uint8_t>(0x7F);
    store_f32(&live[static_cast<std::size_t>(meta_addr(page_bytes, 0, 0, 0, 0))], 1.0e3f);
    const std::vector<QToken> live_out = attention_kernel_shaped(q, cache, live, page_bytes, seq_len, q_len);
    const double d_live = max_abs_diff(clean, live_out);
    measured("max |clean - poisoned-live-row|", d_live);
    check(d_live > 1e-3, "poisoning a live row does move the output, so the mask test is live");

    // Tail boundary: page 2 row 8 is the last valid token (t = 136) and must
    // matter for a single query at end = 137, while row 9 must not.
    std::vector<std::uint8_t> tail8 = pages;
    for (int d = 0; d < D; ++d)
        tail8[static_cast<std::size_t>(k_addr(page_bytes, 2, 8, 0, d))] = static_cast<std::uint8_t>(0x7F);
    const std::vector<QToken> t8 = attention_kernel_shaped(q, cache, tail8, page_bytes, seq_len, q_len);
    const double d_tail8 = max_abs_diff(clean, t8);
    std::vector<std::uint8_t> tail9 = pages;
    for (int d = 0; d < D; ++d)
        tail9[static_cast<std::size_t>(k_addr(page_bytes, 2, 9, 0, d))] = static_cast<std::uint8_t>(0x7F);
    const std::vector<QToken> t9 = attention_kernel_shaped(q, cache, tail9, page_bytes, seq_len, q_len);
    const double d_tail9 = max_abs_diff(clean, t9);
    measured("max shift when page2 row 8 (last valid, t=136) is mutated", d_tail8);
    measured("max shift when page2 row 9 (first invalid, t=137) is mutated", d_tail9);
    check(d_tail8 > 1e-3, "the last valid token is inside the mask boundary");
    check(d_tail9 == 0.0, "the first invalid token is outside the mask boundary");

    // Prefix is mutable too: re-storing a token changes the answer, as a
    // prefix-cache CoW would.
    std::vector<std::uint8_t> rewritten = pages;
    Row other = make_row(rng);
    const KQuant ok = quant_k(other, false);
    const VQuant ov = quant_v(other);
    store_token(rewritten, page_bytes, 5, ok, ov);
    const std::vector<QToken> rw = attention_kernel_shaped(q, cache, rewritten, page_bytes, seq_len, q_len);
    measured("max shift after rewriting token 5", max_abs_diff(clean, rw));
    check(max_abs_diff(clean, rw) > 1e-6, "the page buffer is a live, mutable cache");
}

void test_edges() {
    std::printf("\n== degenerate lengths ==\n");
    const int seq_len = 4, q_len = 1;
    Rng rng(0x1234u);
    std::vector<Row> keys(static_cast<std::size_t>(seq_len)), values(static_cast<std::size_t>(seq_len));
    for (int t = 0; t < seq_len; ++t) {
        keys[static_cast<std::size_t>(t)] = make_row(rng);
        values[static_cast<std::size_t>(t)] = make_row(rng);
    }
    std::vector<QToken> q(1);
    for (int h = 0; h < HQ; ++h) q[0][static_cast<std::size_t>(h)] = make_row(rng);
    const std::vector<CacheRow> empty;
    const std::vector<QToken> out = attention_reference(q, empty, 0, 1);
    check(all_finite(out), "an empty cache yields zeros, not NaN");
    bool zero = true;
    for (const Row& row : out[0])
        for (double v : row)
            if (v != 0.0) zero = false;
    check(zero, "an empty cache yields exactly zero output rows");
    check(attention_reference(q, empty, 0, 0).empty(), "q_len = 0 builds no output rows");
    const std::vector<CacheRow> cache = build_cache(keys, values);
    const std::vector<QToken> one = attention_reference(q, cache, seq_len, q_len);
    check(all_finite(one), "a short cache still produces a finite result");
}

} // namespace

int main() {
    std::printf("K8/V4 host fixture: D=%d GQA=%d hkv=%d (assumed slice) PAGE=%d\n", D, GQA, HKV, PAGE);
    std::printf("slot bytes/head/token = %ld\n", SLOT_BYTES_PER_HEAD);
    test_geometry();
    test_regions();
    test_k_quant();
    test_v_quant();
    test_metadata();
    test_attention();
    test_poison();
    test_edges();
    std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
