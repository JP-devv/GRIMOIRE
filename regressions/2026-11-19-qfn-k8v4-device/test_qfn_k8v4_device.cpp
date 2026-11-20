// SPDX-License-Identifier: GPL-3.0-or-later
//
// PROTOTYPE (not tuning, not a runtime component, not a drop-in):
//
// K8/V4 DENSE-DECODE DEVICE PROTOTYPE -- D=256, one KV head replicated on the
// rank with six query heads (the TP4 per-rank slice), PAGE=64, int8 K, packed
// int4 V, fp32 k_scale / v_scale / v_zero.  The device kernel reads the SAME
// packed page bytes as the host scalar reference in this file, which is a port
// of the fp64 reference path of the host fixture
//   regressions/2026-11-19-qfn-k8v4/test_qfn_k8v4.cpp   (read only, unchanged)
// so a difference can only come from the device path, never from re-quantizing
// or re-packing on the way to the GPU.
//
// The packed layout, the quantizers and the masked-tail validity rule are the
// documented K8/V4 rules; this file reimplements them and re-checks them against
// the numbers the host fixture publishes, so the packer here is proven to be the
// same one before any device comparison is trusted.
//
// Build (JIT, host-side compile only -- see build.sh):
//   icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 ...
//
// Run:
//   build/test_qfn_k8v4_device --host-only        # packer + reference, no queue
//   build/test_qfn_k8v4_device --list-devices      # enumerate GPUs, no queue
//   build/test_qfn_k8v4_device --device <name-substr>   # device run
//
// Exit status: 0 = every check passed, 1 = a check failed, 2 = setup error
// (no device, allocation failure, refused argument, SYCL exception).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ------------------------------------------------------------------ geometry
//
// Documented K8/V4 constants, identical to the host fixture.  The fixture's
// PageLayout only serves hkv in {2, 4}; hkv == 1 (this rank slice) is the same
// extrapolation the fixture makes: 396 B per token per head.
constexpr int D = 256;                                    // head_dim
constexpr int GQA = 6;                                    // Q heads per KV head
constexpr int PAGE = 64;                                   // decode page
constexpr int V4_COLS = D / 2;                             // packed int4 columns
constexpr int SCALE_BYTES = 4;                             // fp32 metadata
constexpr int HKV = 1;                                     // one KV head here
constexpr int HQ = GQA * HKV;                              // 6 Q heads
constexpr int MAX_DECODE_T = PAGE / GQA;                    // 10-token tile
constexpr int SLOT_BYTES = D + V4_COLS + 3 * SCALE_BYTES;   // 396
constexpr int PAGE_BYTES = PAGE * SLOT_BYTES;               // 25344
constexpr int K_BYTES = PAGE * HKV * D;                     // 16384
constexpr int V_BYTES = PAGE * HKV * V4_COLS;               // 8192
constexpr int META_BYTES = PAGE * HKV * SCALE_BYTES;        // 256

static_assert(D == 256, "INV_SQRT_D below is only exact for D = 256");
constexpr float INV_SQRT_D = 1.0f / 16.0f;                  // 1 / sqrt(256)
constexpr double INV_SQRT_D64 = 1.0 / 16.0;
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// Documented page region order: K, V, k_scale, v_scale, v_zero (region-major).
enum class Meta : std::int32_t { k_scale = 0, v_scale = 1, v_zero = 2 };

// Byte offsets inside one page.  These are used by the device kernels and by
// the host packer/reference alike, so both sides address the same bytes with
// the same arithmetic.
inline std::size_t k_off(int page, int tok, int d) {
    return static_cast<std::size_t>(page) * PAGE_BYTES +
           static_cast<std::size_t>(tok) * HKV * D + static_cast<std::size_t>(d);
}

inline std::size_t v_off(int page, int tok, int cell) {
    return static_cast<std::size_t>(page) * PAGE_BYTES + static_cast<std::size_t>(K_BYTES) +
           static_cast<std::size_t>(tok) * HKV * V4_COLS + static_cast<std::size_t>(cell);
}

inline std::size_t m_off(int page, int tok, Meta which) {
    return static_cast<std::size_t>(page) * PAGE_BYTES +
           static_cast<std::size_t>(K_BYTES + V_BYTES) +
           static_cast<std::size_t>(which) * META_BYTES +
           static_cast<std::size_t>(tok) * HKV * SCALE_BYTES;
}

// Little-endian fp32 in the page.  Assembled byte-wise on both host and device
// so no code path depends on a cast or on host endianness.
inline float load_f32(const std::uint8_t* bytes, std::size_t off) {
    const std::uint32_t w = static_cast<std::uint32_t>(bytes[off]) |
                            (static_cast<std::uint32_t>(bytes[off + 1]) << 8) |
                            (static_cast<std::uint32_t>(bytes[off + 2]) << 16) |
                            (static_cast<std::uint32_t>(bytes[off + 3]) << 24);
    return sycl::bit_cast<float>(w);
}

inline void store_f32(std::uint8_t* bytes, std::size_t off, float v) {
    const std::uint32_t w = sycl::bit_cast<std::uint32_t>(v);
    bytes[off + 0] = static_cast<std::uint8_t>(w & 0xFFu);
    bytes[off + 1] = static_cast<std::uint8_t>((w >> 8) & 0xFFu);
    bytes[off + 2] = static_cast<std::uint8_t>((w >> 16) & 0xFFu);
    bytes[off + 3] = static_cast<std::uint8_t>((w >> 24) & 0xFFu);
}

// ------------------------------------------------------------------ reporting
int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

void measured(const std::string& what, double value) {
    std::printf("MEASURED %s = %.12g\n", what.c_str(), value);
}

void derived(const std::string& what, double value) {
    std::printf("DERIVED  %s = %.12g\n", what.c_str(), value);
}

void note(const std::string& what) { std::printf("NOTE     %s\n", what.c_str()); }

// ------------------------------------------------------------------ packer
//
// Documented quantizers, unchanged from the host fixture (which took them from
// the layout/oracle description).  Host only: std::floor is not a device
// operation, and the device never quantizes -- it only decodes these bytes.

using Row = std::array<double, D>;

// Documented rounding: half away from zero.
int round_half_away(double v) {
    const double m = std::floor(std::fabs(v) + 0.5);
    return v < 0.0 ? -static_cast<int>(m) : static_cast<int>(m);
}

struct KQuant {
    std::array<std::int8_t, D> q{};
    float scale = 0.0f;
};

struct VQuant {
    std::array<std::uint8_t, V4_COLS> packed{};
    float scale = 0.0f;
    float zero = 0.0f;
};

// K is symmetric int8: scale = amax/127 (0 for an all-zero row), round half
// away from zero, clamp +-127.
KQuant quant_k(const Row& row) {
    double amax = 0.0;
    for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(row[static_cast<std::size_t>(d)]));
    KQuant out{};
    if (amax == 0.0) return out;
    out.scale = static_cast<float>(amax / 127.0);
    const float inv = 1.0f / out.scale;
    for (int d = 0; d < D; ++d) {
        int q = round_half_away(row[static_cast<std::size_t>(d)] * static_cast<double>(inv));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        out.q[static_cast<std::size_t>(d)] = static_cast<std::int8_t>(q);
    }
    return out;
}

// V is affine int4: scale = max((hi-lo)/15, 1e-8),
// zero = clamp(floor(-lo/scale + 0.5), 0, 15),
// nibble = clamp(floor(v/scale + zero + 0.5), 0, 15), even dim in the low nibble.
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
    for (int cell = 0; cell < V4_COLS; ++cell) {
        std::uint8_t byte = 0;
        for (int half = 0; half < 2; ++half) {
            const int d = 2 * cell + half;  // even dimension is the low nibble
            double q = std::floor(row[static_cast<std::size_t>(d)] /
                                      static_cast<double>(out.scale) + zero + 0.5);
            q = std::min(15.0, std::max(0.0, q));
            const auto nib = static_cast<std::uint8_t>(q);
            byte = static_cast<std::uint8_t>(
                byte | (half == 0 ? nib : static_cast<std::uint8_t>(nib << 4)));
        }
        out.packed[static_cast<std::size_t>(cell)] = byte;
    }
    return out;
}

Row dequant_v(const VQuant& v) {
    Row out{};
    for (int cell = 0; cell < V4_COLS; ++cell) {
        const std::uint8_t byte = v.packed[static_cast<std::size_t>(cell)];
        for (int half = 0; half < 2; ++half) {
            const int nib = half == 0 ? static_cast<int>(byte & 0x0F)
                                      : static_cast<int>(byte >> 4);
            out[static_cast<std::size_t>(2 * cell + half)] =
                (static_cast<double>(nib) - static_cast<double>(v.zero)) *
                static_cast<double>(v.scale);
        }
    }
    return out;
}

// Deterministic rows (SplitMix64), independent of the host fixture's sequence.
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() {  // [-1, 1]
        const std::uint64_t raw = next() >> 11;
        return (static_cast<double>(raw) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
    }
};

Row make_row(Rng& rng) {
    Row row{};
    for (int d = 0; d < D; ++d) row[static_cast<std::size_t>(d)] = rng.uniform();
    return row;
}

// Store one token into the packed page image (hkv = 1).
void pack_token(std::vector<std::uint8_t>& pages, int tok, const KQuant& k, const VQuant& v) {
    const int page = tok / PAGE;
    const int t = tok % PAGE;
    for (int d = 0; d < D; ++d) {
        pages[k_off(page, t, d)] = static_cast<std::uint8_t>(k.q[static_cast<std::size_t>(d)]);
    }
    for (int cell = 0; cell < V4_COLS; ++cell) {
        pages[v_off(page, t, cell)] = v.packed[static_cast<std::size_t>(cell)];
    }
    store_f32(pages.data(), m_off(page, t, Meta::k_scale), k.scale);
    store_f32(pages.data(), m_off(page, t, Meta::v_scale), v.scale);
    store_f32(pages.data(), m_off(page, t, Meta::v_zero), v.zero);
}

// Documented causal lens: index 0 is the longest end, each clamped at 1, and a
// q_len outside the 10-token packed decode tile is refused.
std::vector<std::int32_t> visible_lens(int seq_len, int q_len) {
    if (q_len < 0 || q_len > MAX_DECODE_T) {
        throw std::runtime_error("q_len " + std::to_string(q_len) +
                                 " is outside the 10-token packed decode tile");
    }
    std::vector<std::int32_t> ends(static_cast<std::size_t>(q_len), 0);
    for (int j = 0; j < q_len; ++j) {
        const int end = seq_len - (q_len - 1 - j);
        ends[static_cast<std::size_t>(j)] = end >= 1 ? end : 1;
    }
    return ends;
}

int builtin_q_end(int seq_len, int q_len, int q_tok) { return seq_len - q_len + q_tok + 1; }

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t h = 0xCBF29CE484222325ull;
    for (std::uint8_t b : bytes) {
        h ^= static_cast<std::uint64_t>(b);
        h *= 0x100000001B3ull;
    }
    return h;
}

// ------------------------------------------------------------------ the case
struct Case {
    std::string name;
    int seq_len = 0;
    int q_len = 0;
    std::vector<std::uint8_t> pages;     // n_pages() * PAGE_BYTES, the only KV source
    std::vector<std::int8_t> q_codes;    // q_len * HQ * D
    std::vector<float> q_scale;          // q_len * HQ
    std::vector<std::int32_t> ends;      // q_len
    int n_pages() const { return seq_len <= 0 ? 0 : (seq_len + PAGE - 1) / PAGE; }
    int pairs() const { return q_len * HQ; }
};

Case build_case(const std::string& name, int seq_len, int q_len, std::uint64_t seed) {
    if (seq_len < 0) throw std::runtime_error("negative seq_len");
    Case c;
    c.name = name;
    c.seq_len = seq_len;
    c.q_len = q_len;
    c.ends = visible_lens(seq_len, q_len);  // refuses q_len > 10
    c.pages.assign(static_cast<std::size_t>(c.n_pages()) * PAGE_BYTES, 0);
    Rng rng(seed);
    for (int t = 0; t < seq_len; ++t) {
        const Row krow = make_row(rng);
        const Row vrow = make_row(rng);
        pack_token(c.pages, t, quant_k(krow), quant_v(vrow));
    }
    c.q_codes.assign(static_cast<std::size_t>(c.pairs()) * D, 0);
    c.q_scale.assign(static_cast<std::size_t>(c.pairs()), 0.0f);
    for (int j = 0; j < q_len; ++j) {
        for (int h = 0; h < HQ; ++h) {
            const KQuant qq = quant_k(make_row(rng));
            const std::size_t pair = static_cast<std::size_t>(j) * HQ + static_cast<std::size_t>(h);
            c.q_scale[pair] = qq.scale;
            for (int d = 0; d < D; ++d) {
                c.q_codes[pair * D + static_cast<std::size_t>(d)] = qq.q[static_cast<std::size_t>(d)];
            }
        }
    }
    return c;
}

// ------------------------------------------------------------------ reference
//
// fp64 scalar port of the host fixture's reference path, but every K and V
// value is decoded from the packed page bytes of `c`, so this and the kernel
// consume one identical buffer.  Masked tokens (tok >= n_valid or k_abs >= end)
// are skipped, as the fixture's reference skips them.
struct RefOut {
    std::vector<double> attn;    // pairs * D
    std::vector<double> scores;  // pairs * PAGE * n_pages, masked entries = -inf
    std::vector<double> vdeq;    // seq_len * D, decoded V rows
};

RefOut run_reference(const Case& c) {
    RefOut out;
    const int np = c.n_pages();
    const std::size_t nscores = static_cast<std::size_t>(c.pairs()) * PAGE *
                                static_cast<std::size_t>(np);
    out.attn.assign(static_cast<std::size_t>(c.pairs()) * D, 0.0);
    out.scores.assign(nscores, -std::numeric_limits<double>::infinity());
    out.vdeq.assign(static_cast<std::size_t>(c.seq_len) * D, 0.0);

    // Decode V once from the packed bytes (int4 affine, even dim low nibble).
    for (int t = 0; t < c.seq_len; ++t) {
        const int page = t / PAGE;
        const int tok = t % PAGE;
        const float vs = load_f32(c.pages.data(), m_off(page, tok, Meta::v_scale));
        const float vz = load_f32(c.pages.data(), m_off(page, tok, Meta::v_zero));
        for (int cell = 0; cell < V4_COLS; ++cell) {
            const std::uint8_t byte = c.pages[v_off(page, tok, cell)];
            for (int half = 0; half < 2; ++half) {
                const int nib = half == 0 ? static_cast<int>(byte & 0x0F)
                                         : static_cast<int>(byte >> 4);
                out.vdeq[static_cast<std::size_t>(t) * D +
                         static_cast<std::size_t>(2 * cell + half)] =
                    (static_cast<double>(nib) - static_cast<double>(vz)) *
                    static_cast<double>(vs);
            }
        }
    }

    const std::size_t per_pair = static_cast<std::size_t>(np) * PAGE;
    std::vector<double> p(per_pair, 0.0);
    for (int pair = 0; pair < c.pairs(); ++pair) {
        const int j = pair / HQ;
        const int end = c.ends[static_cast<std::size_t>(j)];
        const std::size_t qbase = static_cast<std::size_t>(pair) * D;
        const double qs = static_cast<double>(c.q_scale[static_cast<std::size_t>(pair)]);
        double run_max = -std::numeric_limits<double>::infinity();
        for (int page = 0; page < np; ++page) {
            const int n_valid = std::max(0, std::min(PAGE, c.seq_len - page * PAGE));
            for (int tok = 0; tok < PAGE; ++tok) {
                const std::size_t sidx = (static_cast<std::size_t>(pair) * np +
                                          static_cast<std::size_t>(page)) *
                                             PAGE + static_cast<std::size_t>(tok);
                const long long k_abs = static_cast<long long>(page) * PAGE + tok;
                if (tok >= n_valid || k_abs >= end) continue;  // stays -inf
                const float ks = load_f32(c.pages.data(), m_off(page, tok, Meta::k_scale));
                int acc8 = 0;
                for (int d = 0; d < D; ++d) {
                    const auto kv = static_cast<std::int8_t>(c.pages[k_off(page, tok, d)]);
                    acc8 += static_cast<int>(c.q_codes[qbase + static_cast<std::size_t>(d)]) *
                            static_cast<int>(kv);
                }
                const double s = static_cast<double>(acc8) * qs * static_cast<double>(ks) *
                                 INV_SQRT_D64;
                out.scores[sidx] = s;
                run_max = std::max(run_max, s);
            }
        }
        if (run_max == -std::numeric_limits<double>::infinity()) continue;  // zeros

        // One stable softmax over every visible token of every page (the
        // fixture's reference has no page split at all), then PV in fp64.
        const std::size_t base_idx = static_cast<std::size_t>(pair) * per_pair;
        double total = 0.0;
        for (std::size_t k = 0; k < per_pair; ++k) {
            const double e = std::exp(out.scores[base_idx + k] - run_max);
            p[k] = e;
            total += e;
        }
        if (total == 0.0) continue;
        const double inv = 1.0 / total;
        for (std::size_t k = 0; k < per_pair; ++k) {
            const double pr = p[k] * inv;
            if (pr == 0.0) continue;
            const int page = static_cast<int>(k / PAGE);
            const int tok = static_cast<int>(k % PAGE);
            const int t = page * PAGE + tok;
            if (t >= c.seq_len) continue;
            for (int d = 0; d < D; ++d) {
                out.attn[qbase + static_cast<std::size_t>(d)] +=
                    pr * out.vdeq[static_cast<std::size_t>(t) * D + static_cast<std::size_t>(d)];
            }
        }
    }
    return out;
}

// ------------------------------------------------------------------ USM guard
template <class T>
class DevBuf {
  public:
    DevBuf(sycl::queue& q, std::size_t count)
        : q_(q), count_(count == 0 ? 1 : count),
          ptr_(sycl::malloc_device<T>(count == 0 ? 1 : count, q)) {
        if (ptr_ == nullptr) throw std::runtime_error("sycl::malloc_device returned null");
    }
    ~DevBuf() {
        if (ptr_ != nullptr) sycl::free(ptr_, q_);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t count() const noexcept { return count_; }
    void upload(const std::vector<T>& src) {
        if (src.empty()) return;
        if (src.size() > count_) throw std::runtime_error("upload exceeds allocation");
        q_.memcpy(ptr_, src.data(), src.size() * sizeof(T)).wait_and_throw();
    }
    void fill_zero() {
        q_.memset(ptr_, 0, count_ * sizeof(T)).wait_and_throw();
    }
    void download(std::vector<T>& dst) const {
        if (dst.empty()) return;
        if (dst.size() > count_) throw std::runtime_error("download exceeds allocation");
        q_.memcpy(dst.data(), ptr_, dst.size() * sizeof(T)).wait_and_throw();
    }

  private:
    sycl::queue& q_;
    std::size_t count_;
    T* ptr_;
};

// Kernel names (JIT: distinct names keep the device images separate).
struct ScoreKernel;
struct SoftmaxKernel;
struct PvKernel;
struct MergeKernel;

// ------------------------------------------------------------------ device
struct DeviceOut {
    std::vector<float> attn;    // pairs * D
    std::vector<float> scores;  // pairs * PAGE * n_pages, masked = -inf
};

// Dense decode, one call per q_len (<= 10) query tokens over n_pages pages:
//   score  : int8 dot for all 64 page tokens, then masked to -inf
//   softmax: per-page max / sum, probabilities rounded through fp16
//   pv     : per-page PV partial over fp16 probabilities, fp32 accumulator
//   merge  : running M/L rescale across pages, fp32
// This mirrors the documented paged-decode split (S1 + S2) rather than inventing
// a different numerics; nothing here is tuned.
DeviceOut run_device(sycl::queue& q, const Case& c) {
    const int pairs = c.pairs();
    const int np = c.n_pages();
    DeviceOut out;
    out.attn.assign(static_cast<std::size_t>(pairs) * D, 0.0f);
    out.scores.assign(static_cast<std::size_t>(pairs) * PAGE * static_cast<std::size_t>(np), 0.0f);
    if (pairs == 0) return out;

    const std::size_t npages = static_cast<std::size_t>(np);
    const std::size_t nscores = static_cast<std::size_t>(pairs) * PAGE * npages;

    DevBuf<std::int8_t> dq(q, c.q_codes.size());
    dq.upload(c.q_codes);
    DevBuf<float> dqs(q, c.q_scale.size());
    dqs.upload(c.q_scale);
    DevBuf<std::int32_t> dends(q, c.ends.size());
    dends.upload(c.ends);
    DevBuf<std::uint8_t> dpages(q, c.pages.size());
    dpages.upload(c.pages);
    DevBuf<float> dscore(q, nscores);
    DevBuf<sycl::half> dprob(q, nscores);
    DevBuf<float> dpm(q, static_cast<std::size_t>(pairs) * npages);
    DevBuf<float> dps(q, static_cast<std::size_t>(pairs) * npages);
    DevBuf<float> dpacc(q, static_cast<std::size_t>(pairs) * npages * D);
    DevBuf<float> dattn(q, static_cast<std::size_t>(pairs) * D);
    if (np == 0) {
        return out;  // empty cache: exactly zero rows, no kernels, no garbage
    }

    const std::int8_t* q8 = dq.get();
    const float* qsc = dqs.get();
    const std::int32_t* ends = dends.get();
    const std::uint8_t* pages = dpages.get();
    float* score = dscore.get();
    sycl::half* prob = dprob.get();
    float* pm = dpm.get();
    float* ps = dps.get();
    float* pacc = dpacc.get();
    float* attn = dattn.get();
    const int seq_len = c.seq_len;

    // 1) scores: one work-group of D lanes per (pair, page, token).
    const int score_groups = pairs * PAGE * np;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ScoreKernel>(
            sycl::nd_range<1>(static_cast<std::size_t>(score_groups) * D, D),
            [=](sycl::nd_item<1> it) {
                const std::size_t g = it.get_global_id(0);
                const std::size_t grp = g / static_cast<std::size_t>(D);
                const int lane = static_cast<int>(it.get_local_id(0));
                const int tok = static_cast<int>(grp % static_cast<std::size_t>(PAGE));
                const std::size_t rest = grp / static_cast<std::size_t>(PAGE);
                const int page = static_cast<int>(rest % npages);
                const int pair = static_cast<int>(rest / npages);
                const std::size_t qbase = static_cast<std::size_t>(pair) * D;
                const auto kbyte = static_cast<std::int8_t>(pages[k_off(page, tok, lane)]);
                const std::int32_t prod =
                    static_cast<std::int32_t>(q8[qbase + static_cast<std::size_t>(lane)]) *
                    static_cast<std::int32_t>(kbyte);
                const std::int32_t acc =
                    sycl::reduce_over_group(it.get_group(), prod, sycl::plus<>());
                if (lane != 0) return;
                // Documented kernel: read k_scale and form the score for every
                // one of the 64 page tokens, and only then apply the mask.
                const float ks = load_f32(pages, m_off(page, tok, Meta::k_scale));
                float s = static_cast<float>(acc) * qsc[pair] * ks * INV_SQRT_D;
                const int n_valid = seq_len - page * PAGE > PAGE ? PAGE
                                  : (seq_len - page * PAGE < 0 ? 0 : seq_len - page * PAGE);
                const long long k_abs = static_cast<long long>(page) * PAGE + tok;
                if (tok >= n_valid || k_abs >= static_cast<long long>(ends[pair / HQ])) {
                    s = NEG_INF;
                }
                score[grp * PAGE + static_cast<std::size_t>(tok)] = s;
            });
    });

    // 2) per-page softmax (max-shifted) with fp16 probabilities.
    const std::size_t softmax_blocks = static_cast<std::size_t>(pairs) * npages;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<SoftmaxKernel>(
            sycl::nd_range<1>(softmax_blocks * PAGE, PAGE),
            [=](sycl::nd_item<1> it) {
                const std::size_t block = it.get_global_id(0) / static_cast<std::size_t>(PAGE);
                const std::size_t tok = it.get_local_id(0);
                const float s = score[block * PAGE + tok];
                const float m = sycl::reduce_over_group(it.get_group(), s, sycl::maximum<>());
                const float e = (m == NEG_INF) ? 0.0f : sycl::exp(s - m);
                const float total =
                    sycl::reduce_over_group(it.get_group(), e, sycl::plus<>());
                prob[block * PAGE + tok] =
                    (total > 0.0f) ? sycl::half(e / total) : sycl::half(0.0f);
                if (tok == 0) {
                    pm[block] = m;
                    ps[block] = total;
                }
            });
    });

    // 3) per-page PV partial; one lane per output dimension.
    q.submit([&](sycl::handler& h) {
        h.parallel_for<PvKernel>(
            sycl::nd_range<1>(softmax_blocks * D, D),
            [=](sycl::nd_item<1> it) {
                const std::size_t block = it.get_global_id(0) / static_cast<std::size_t>(D);
                const int d = static_cast<int>(it.get_local_id(0));
                const int page = static_cast<int>(block % npages);
                const int cell = d >> 1;
                const int shift = d & 1;
                const int n_valid = seq_len - page * PAGE > PAGE ? PAGE
                                  : (seq_len - page * PAGE < 0 ? 0 : seq_len - page * PAGE);
                float acc = 0.0f;
                for (int tok = 0; tok < PAGE; ++tok) {
                    // Guard under test: a row outside the page's valid length
                    // must not reach the affine metadata or the nibbles.
                    if (tok >= n_valid) continue;
                    const float pr = static_cast<float>(prob[block * PAGE +
                                                             static_cast<std::size_t>(tok)]);
                    if (pr == 0.0f) continue;
                    const float vs = load_f32(pages, m_off(page, tok, Meta::v_scale));
                    const float vz = load_f32(pages, m_off(page, tok, Meta::v_zero));
                    const std::uint8_t byte = pages[v_off(page, tok, cell)];
                    const int nib = (shift == 0) ? static_cast<int>(byte & 0x0F)
                                                 : static_cast<int>(byte >> 4);
                    const sycl::half vb =
                        sycl::half((static_cast<float>(nib) - vz) * vs);
                    acc += pr * static_cast<float>(vb);
                }
                pacc[block * D + static_cast<std::size_t>(d)] = acc;
            });
    });

    // 4) cross-page merge with running max / running normaliser (fp32).
    q.submit([&](sycl::handler& h) {
        h.parallel_for<MergeKernel>(sycl::range<1>(static_cast<std::size_t>(pairs)),
                                    [=](sycl::id<1> idx) {
            const std::size_t pair = idx[0];
            float acc[D] = {};
            float run_m = NEG_INF;
            float run_l = 0.0f;
            for (std::size_t page = 0; page < npages; ++page) {
                const int p = static_cast<int>(page);
                const int n_valid = seq_len - p * PAGE > PAGE ? PAGE
                                  : (seq_len - p * PAGE < 0 ? 0 : seq_len - p * PAGE);
                if (n_valid <= 0) continue;
                const std::size_t block = pair * npages + page;
                const float m = pm[block];
                if (m == NEG_INF) continue;
                const float total = ps[block];
                const float new_m = sycl::fmax(run_m, m);
                const float old_scale = (run_m == NEG_INF) ? 0.0f : sycl::exp(run_m - new_m);
                const float weight = sycl::exp(m - new_m) * total;
                for (int d = 0; d < D; ++d) {
                    acc[d] = acc[d] * old_scale +
                             pacc[block * D + static_cast<std::size_t>(d)] * weight;
                }
                run_l = run_l * old_scale + weight;
                run_m = new_m;
            }
            const float norm = (run_l == 0.0f) ? 0.0f : 1.0f / run_l;
            for (int d = 0; d < D; ++d) {
                attn[pair * D + static_cast<std::size_t>(d)] = acc[d] * norm;
            }
        });
    });

    q.wait_and_throw();
    dattn.download(out.attn);
    dscore.download(out.scores);
    return out;
}

// ------------------------------------------------------------------ compare
double max_abs_diff(const std::vector<float>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, d);
    }
    return worst;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, d);
    }
    return worst;
}

// Score-buffer comparison that understands the -inf mask: two masked entries
// agree, a masked/unmasked disagreement is a hard failure.
double max_abs_diff_scores(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("score comparison size mismatch");
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool fa = std::isfinite(a[i]);
        const bool fb = std::isfinite(b[i]);
        if (!fa && !fb) continue;
        if (!fa || !fb) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, d);
    }
    return worst;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

bool all_finite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

double max_abs_value(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// max |device score - fp64 score| over tokens the mask keeps.
double score_diff(const std::vector<float>& dev, const std::vector<double>& ref) {
    if (dev.size() != ref.size()) throw std::runtime_error("score comparison size mismatch");
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i])) continue;  // masked in the reference too
        if (!std::isfinite(dev[i])) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, std::fabs(static_cast<double>(dev[i]) - ref[i]));
    }
    return worst;
}

// ------------------------------------------------------------------ host tests
void test_layout() {
    std::printf("\n== geometry and page layout (prototype constants) ==\n");
    measured("SLOT_BYTES (D + V4_COLS + 3*4)", static_cast<double>(SLOT_BYTES));
    measured("PAGE_BYTES", static_cast<double>(PAGE_BYTES));
    check(SLOT_BYTES == 396, "a token head slot is 396 B");
    check(PAGE_BYTES == 25344, "a page is 64 slots = 25344 B");
    check(HQ == 6, "one KV head carries exactly 6 query heads");
    check(MAX_DECODE_T == 10, "the packed decode tile is 10 tokens");

    check(m_off(0, 0, Meta::k_scale) == 24576, "k_scale starts at 24576");
    check(m_off(0, 0, Meta::v_scale) == 24832, "v_scale starts at 24832");
    check(m_off(0, 0, Meta::v_zero) == 25088, "v_zero starts at 25088");
    check(m_off(0, 63, Meta::v_zero) + SCALE_BYTES == PAGE_BYTES,
          "the last v_zero closes the page at 25344");
    check(k_off(0, 1, 0) == 256, "K token stride is hkv*D = 256 B");
    check(v_off(0, 0, 0) == K_BYTES, "V starts right after the K region");
    check(v_off(0, 1, 0) - v_off(0, 0, 0) == 128, "V token stride is 128 B");
    check(k_off(1, 0, 0) == PAGE_BYTES, "the page stride is the whole page");

    // fp32 metadata must survive bit for bit (why the region is 4 B, not 2).
    std::vector<std::uint8_t> page(static_cast<std::size_t>(PAGE_BYTES), 0);
    const float tiny = 1.0e-7f;
    store_f32(page.data(), m_off(0, 7, Meta::k_scale), tiny);
    check(load_f32(page.data(), m_off(0, 7, Meta::k_scale)) == tiny,
          "a 1e-7 scale round-trips through the page byte for byte");
}

void test_packer() {
    std::printf("\n== packer cross-check against the numbers the host fixture publishes ==\n");

    // Tie row: amax = 127 makes the K scale exactly 1.0, so the documented
    // half-away rule and a ties-to-even rule disagree.
    Row ties{};
    ties[0] = 127.0;
    ties[1] = 0.5;
    ties[2] = -0.5;
    ties[3] = 2.5;
    ties[4] = 1.5;
    const KQuant tie = quant_k(ties);
    const int expect[5] = {127, 1, -1, 3, 2};
    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        if (static_cast<int>(tie.q[static_cast<std::size_t>(i)]) != expect[i]) ok = false;
    }
    std::printf("ROUNDING tie row q = [%d %d %d %d %d]\n",
                static_cast<int>(tie.q[0]), static_cast<int>(tie.q[1]),
                static_cast<int>(tie.q[2]), static_cast<int>(tie.q[3]),
                static_cast<int>(tie.q[4]));
    check(tie.scale == 1.0f, "amax = 127 gives a K scale of exactly 1.0");
    check(ok, "documented half-away rounding gives [127 1 -1 3 2]");

    // V ladder: byte = even | odd << 4.
    Row ladder{};
    for (int d = 0; d < D; ++d) ladder[static_cast<std::size_t>(d)] = static_cast<double>(d % 16);
    const VQuant lv = quant_v(ladder);
    check(lv.scale == 1.0f && lv.zero == 0.0f, "lo=0 hi=15 gives V scale 1, zero 0");
    check(lv.packed[0] == 0x10, "byte 0 packs dims (0,1) as 0x10 (even dim low nibble)");
    bool packing_ok = true;
    for (int cell = 0; cell < V4_COLS; ++cell) {
        const int even = (2 * cell) % 16;
        const int odd = (2 * cell + 1) % 16;
        if (lv.packed[static_cast<std::size_t>(cell)] !=
            static_cast<std::uint8_t>(even | (odd << 4))) {
            packing_ok = false;
        }
    }
    check(packing_ok, "all 128 packed bytes follow even | odd << 4");
    const Row lback = dequant_v(lv);
    double lerr = 0.0;
    for (int d = 0; d < D; ++d) {
        lerr = std::max(lerr, std::fabs(lback[static_cast<std::size_t>(d)] -
                                        ladder[static_cast<std::size_t>(d)]));
    }
    check(lerr == 0.0, "the integer ladder dequantizes exactly");

    // Flat positive / flat negative rows saturate at the 1e-8 scale floor.
    Row cpos{};
    for (int d = 0; d < D; ++d) cpos[static_cast<std::size_t>(d)] = 0.5;
    const VQuant cp = quant_v(cpos);
    const Row cpback = dequant_v(cp);
    check(cp.scale == static_cast<float>(1e-8) && cp.zero == 0.0f && cp.packed[0] == 0xFF,
          "flat +0.5 hits the scale floor, zero 0 and 0xFF nibbles");
    check(std::fabs(cpback[0] - 1.5e-7) < 1e-12, "flat +0.5 dequantizes to 1.5e-7, not 0.5");
    Row cneg{};
    for (int d = 0; d < D; ++d) cneg[static_cast<std::size_t>(d)] = -0.5;
    const VQuant cn = quant_v(cneg);
    const Row cnback = dequant_v(cn);
    check(cn.zero == 15.0f && cn.packed[0] == 0x00,
          "flat -0.5 clamps the zero point to 15 and every nibble to 0");
    check(std::fabs(cnback[0] + 1.5e-7) < 1e-12, "flat -0.5 dequantizes to -1.5e-7");

    // Documented open edge case: an all-negative spread row collapses.
    Row neg{-90.0, -100.0};
    for (int d = 2; d < D; ++d) neg[static_cast<std::size_t>(d)] = -90.0 - (d % 11);
    const VQuant nv = quant_v(neg);
    int codes = 0;
    for (int q = 0; q < 16; ++q) {
        bool used = false;
        for (int cell = 0; cell < V4_COLS; ++cell) {
            const std::uint8_t byte = nv.packed[static_cast<std::size_t>(cell)];
            if (static_cast<int>(byte & 0x0F) == q || static_cast<int>(byte >> 4) == q) used = true;
        }
        if (used) ++codes;
    }
    check(nv.zero == 15.0f, "an all-negative row clamps its zero point to 15");
    check(codes == 1, "only one of the 16 V codes survives on that row");

    // Zero K row: scale 0 and codes 0, no NaN anywhere in the decode path.
    const KQuant zk = quant_k(Row{});
    check(zk.scale == 0.0f, "an all-zero K row gets scale 0");

    // Lens: causal ends and the 10-token refusal.
    const std::vector<std::int32_t> e137 = visible_lens(137, 3);
    std::printf("LENS     seq=137 q_len=3 ends = [%d %d %d]\n", e137[0], e137[1], e137[2]);
    check(e137[0] == 135 && e137[1] == 136 && e137[2] == 137,
          "seq 137 / q_len 3 gives ends [135 136 137]");
    bool agree = true;
    for (int j = 0; j < 3; ++j) {
        if (e137[static_cast<std::size_t>(j)] != builtin_q_end(137, 3, j)) agree = false;
    }
    check(agree, "the lens agrees with the builtin q_end formula");
    const std::vector<std::int32_t> e1 = visible_lens(1, 3);
    check(e1[0] == 1 && e1[1] == 1 && e1[2] == 1, "ends clamp at 1 on a short cache");
    bool refused = false;
    try {
        (void)visible_lens(137, MAX_DECODE_T + 1);
    } catch (const std::runtime_error&) {
        refused = true;
    }
    check(refused, "a q_len above the 10-token decode tile is refused");
}

// ------------------------------------------------------------------ device run
sycl::queue make_queue(const std::string& device_filter, bool use_cpu) {
    const std::vector<sycl::device> all = sycl::device::get_devices();
    std::vector<sycl::device> candidates;
    for (const sycl::device& dev : all) {
        const bool wanted = use_cpu ? dev.is_cpu() : dev.is_gpu();
        if (!wanted) continue;
        const std::string name = dev.get_info<sycl::info::device::name>();
        std::printf("DEVICE   candidate: %s  [%s]\n", name.c_str(),
                    use_cpu ? "cpu" : "gpu");
        if (device_filter.empty() || name.find(device_filter) != std::string::npos) {
            candidates.push_back(dev);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error(
            use_cpu ? "no SYCL CPU device (the CPU device path is optional)"
                    : "no SYCL GPU device; refusing to run the device path "
                      "(use --host-only for the reference only)");
    }
    sycl::queue q{candidates.front(), sycl::property::queue::in_order{}};
    std::printf("DEVICE   using: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("DRIVER   %s\n",
                q.get_device()
                    .get_info<sycl::info::device::driver_version>()
                    .c_str());
    return q;
}

void list_devices() {
    for (const sycl::device& dev : sycl::device::get_devices()) {
        std::printf("DEVICE   %s  [%s]  driver %s\n",
                    dev.get_info<sycl::info::device::name>().c_str(),
                    dev.is_gpu() ? "gpu" : "cpu",
                    dev.get_info<sycl::info::device::driver_version>().c_str());
    }
}

struct CaseSpec {
    const char* name;
    int seq_len;
    int q_len;
    std::uint64_t seed;
};

// One device case end to end: hash, finiteness, reference comparison, score
// comparison.  Returns the device result so the controls can reuse it.
DeviceOut run_and_compare(sycl::queue& q, const Case& c) {
    const std::uint64_t hash_before = fnv1a(c.pages);
    std::printf("\n-- case %s: seq_len=%d q_len=%d pages=%d bytes=%zu hash=%016llx\n",
                c.name.c_str(), c.seq_len, c.q_len, c.n_pages(), c.pages.size(),
                static_cast<unsigned long long>(hash_before));

    const RefOut ref = run_reference(c);
    const DeviceOut dev = run_device(q, c);

    check(fnv1a(c.pages) == hash_before, c.name + ": the packed bytes were not mutated by the run");
    check(all_finite(ref.attn), c.name + ": fp64 reference output is finite");
    check(all_finite(dev.attn), c.name + ": device output is finite");
    measured(c.name + ": max |reference|", max_abs_value(ref.attn));
    const double d = max_abs_diff(dev.attn, ref.attn);
    measured(c.name + ": max |device - reference|", d);
    derived(c.name + ": fp16-probability / fp16-V boundary on |v|<=1", 5e-3);
    check(d < 5e-3, c.name + ": device matches the packed-byte fp64 reference inside the bound");

    const double ds = score_diff(dev.scores, ref.scores);
    measured(c.name + ": max |device score - fp64 score| on visible tokens", ds);
    derived(c.name + ": int8 dot is exact, fp32 vs fp64 scale multiply", 1e-4);
    check(ds < 1e-4, c.name + ": the int8 dot and score scaling agree on every visible token");
    return dev;
}

// Copy of a case with one row's bytes clobbered.  Invalid (masked) rows get
// NaN / inf metadata too, as the fixture's poison test does.
enum class Poison { data_only, affine_nan };

Case poisoned_copy(const Case& base, int tok, Poison kind) {
    Case c = base;
    const int page = tok / PAGE;
    const int t = tok % PAGE;
    for (int d = 0; d < D; ++d) c.pages[k_off(page, t, d)] = static_cast<std::uint8_t>(0x7F);
    for (int cell = 0; cell < V4_COLS; ++cell) {
        c.pages[v_off(page, t, cell)] = static_cast<std::uint8_t>(0xFF);
    }
    if (kind == Poison::affine_nan) {
        store_f32(c.pages.data(), m_off(page, t, Meta::k_scale),
                  std::numeric_limits<float>::quiet_NaN());
        store_f32(c.pages.data(), m_off(page, t, Meta::v_scale),
                  std::numeric_limits<float>::infinity());
        store_f32(c.pages.data(), m_off(page, t, Meta::v_zero), 1.0e30f);
    } else {
        store_f32(c.pages.data(), m_off(page, t, Meta::k_scale), 1.0e3f);
    }
    return c;
}

void shift_report(const std::string& what, double shift, double floor) {
    measured(what, shift);
    check(shift > floor, what + ": the mutation is visible (control is live)");
}

void zero_report(const std::string& what, double shift) {
    measured(what, shift);
    check(shift == 0.0, what + ": the mutation changes exactly nothing");
}

// ------------------------------------------------------------------ main
int run(int argc, char** argv) {
    bool host_only = false;
    bool want_list = false;
    bool use_cpu = false;
    std::string device_filter;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host-only") {
            host_only = true;
        } else if (arg == "--list-devices") {
            want_list = true;
        } else if (arg == "--cpu") {
            use_cpu = true;
        } else if (arg == "--device") {
            if (i + 1 >= argc) throw std::runtime_error("--device needs a name substring");
            device_filter = argv[++i];
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    std::printf("PROTOTYPE K8/V4 dense decode, device vs packed-byte fp64 reference\n");
    std::printf("geometry: D=%d GQA=%d HQ=%d PAGE=%d V4_COLS=%d page=%d B\n", D, GQA, HQ, PAGE,
                V4_COLS, PAGE_BYTES);
    note("prototype only: no runtime code is used or linked, nothing is tuned");

    if (want_list) {
        list_devices();
        return 0;
    }

    test_layout();
    test_packer();

    const std::array<CaseSpec, 7> specs{{
        {"seq137_q3 (3 pages, crossing)", 137, 3, 0xA771u},
        {"seq64_q1 (one full page)", 64, 1, 0x64u},
        {"seq65_q2 (crosses the boundary by one token)", 65, 2, 0x65u},
        {"seq128_q1 (two full pages)", 128, 1, 0x128u},
        {"seq192_q10 (max decode tile, 3 full pages)", 192, 10, 0x192u},
        {"seq1_q1 (single token)", 1, 1, 0x1u},
        {"seq0_q1 (empty cache)", 0, 1, 0x0u},
    }};

    std::vector<Case> cases;
    cases.reserve(specs.size());
    for (const CaseSpec& spec : specs) {
        cases.push_back(build_case(spec.name, spec.seq_len, spec.q_len, spec.seed));
    }

    if (host_only) {
        std::printf("\n== host-only: packer + fp64 reference, no queue, no device memory ==\n");
        for (const Case& c : cases) {
            std::printf("CASE     %s seq=%d q_len=%d pages=%d hash=%016llx\n", c.name.c_str(),
                        c.seq_len, c.q_len, c.n_pages(),
                        static_cast<unsigned long long>(fnv1a(c.pages)));
            const RefOut ref = run_reference(c);
            check(all_finite(ref.attn), c.name + ": reference output is finite");
            measured(c.name + ": max |reference|", max_abs_value(ref.attn));
        }
        // Single-token identity is a reference-only property: one visible token
        // returns that token's decoded V row.
        if (!cases.empty()) {
            const Case& one = cases[5];
            const RefOut ref = run_reference(one);
            const int t = 0;
            const int page = t / PAGE;
            const int tok = t % PAGE;
            const float vs = load_f32(one.pages.data(), m_off(page, tok, Meta::v_scale));
            const float vz = load_f32(one.pages.data(), m_off(page, tok, Meta::v_zero));
            double worst = 0.0;
            for (int pair = 0; pair < one.pairs(); ++pair) {
                for (int d = 0; d < D; ++d) {
                    const std::uint8_t byte = one.pages[v_off(page, tok, d >> 1)];
                    const int nib = (d & 1) ? static_cast<int>(byte >> 4)
                                            : static_cast<int>(byte & 0x0F);
                    const double v = (static_cast<double>(nib) - static_cast<double>(vz)) *
                                     static_cast<double>(vs);
                    worst = std::max(worst,
                                     std::fabs(ref.attn[static_cast<std::size_t>(pair) * D +
                                                        static_cast<std::size_t>(d)] - v));
                }
            }
            measured("seq1: max |reference - decoded V row|", worst);
            check(worst < 1e-12, "a one-token cache returns the decoded V row itself");
        }
        std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK (host only)" : "FAILED",
                    g_failures);
        return g_failures == 0 ? 0 : 1;
    }

    sycl::queue q = make_queue(device_filter, use_cpu);

    // Per-case comparison against the fp64 reference over identical bytes.
    std::vector<DeviceOut> dev;
    dev.reserve(cases.size());
    for (const Case& c : cases) dev.push_back(run_and_compare(q, c));

    // ---- controls, on the 3-page crossing case ------------------------------
    std::printf("\n== controls on %s ==\n", cases[0].name.c_str());
    const Case& base = cases[0];
    const DeviceOut& clean = dev[0];
    const int last_page = base.n_pages() - 1;
    const int n_valid_last = std::min(PAGE, base.seq_len - last_page * PAGE);
    std::printf("VALIDITY pages of %d tokens: n_valid(last page) = %d\n", base.seq_len,
                n_valid_last);
    check(base.n_pages() == 3 && n_valid_last == 9, "137 tokens give pages of 64/64/9");
    check(base.ends[0] == 135 && base.ends[1] == 136 && base.ends[2] == 137,
          "the q_len=3 lens ends are [135 136 137]");

    // 1) The masked tail of the last page: K bytes, nibbles and all three fp32
    //    affine tensors are garbage.  Nothing may move, and nothing may be NaN.
    const Case tail = poisoned_copy(base, last_page * PAGE + n_valid_last, Poison::affine_nan);
    {
        Case tail_all = tail;
        for (int tok = last_page * PAGE + n_valid_last + 1; tok < (last_page + 1) * PAGE; ++tok) {
            const int page = tok / PAGE;
            const int t = tok % PAGE;
            for (int d = 0; d < D; ++d)
                tail_all.pages[k_off(page, t, d)] = static_cast<std::uint8_t>(0x7F);
            for (int cell = 0; cell < V4_COLS; ++cell)
                tail_all.pages[v_off(page, t, cell)] = static_cast<std::uint8_t>(0xFF);
            store_f32(tail_all.pages.data(), m_off(page, t, Meta::k_scale),
                      std::numeric_limits<float>::quiet_NaN());
            store_f32(tail_all.pages.data(), m_off(page, t, Meta::v_scale),
                      std::numeric_limits<float>::infinity());
            store_f32(tail_all.pages.data(), m_off(page, t, Meta::v_zero), 1.0e30f);
        }
        const DeviceOut poi = run_device(q, tail_all);
        check(all_finite(poi.attn), "a NaN/inf poisoned masked tail leaves the device output finite");
        zero_report("max |clean - poisoned-masked-tail| (device)", max_abs_diff(clean.attn, poi.attn));
        zero_report("max |clean - poisoned-masked-tail| (scores, device)",
                    max_abs_diff_scores(clean.scores, poi.scores));
        const RefOut ref_clean = run_reference(base);
        const RefOut ref_poi = run_reference(tail_all);
        zero_report("max |clean - poisoned-masked-tail| (fp64 reference)",
                    max_abs_diff(ref_clean.attn, ref_poi.attn));
    }

    // 2) Negative control: poison a LIVE row.  If this changed nothing, the
    //    masked-tail check above would prove nothing.
    {
        const Case live = poisoned_copy(base, 0, Poison::affine_nan);
        const DeviceOut poi = run_device(q, live);
        check(all_finite(poi.attn), "poisoning a live row keeps the device output finite");
        shift_report("max |clean - poisoned-live-row| (device)", max_abs_diff(clean.attn, poi.attn),
                     1e-3);
    }

    // 3) Mask boundary from both sides: page 2 row 8 is token 136 (last valid,
    //    visible only to the last query) and row 9 is token 137 (invalid).
    {
        const Case t136 = poisoned_copy(base, last_page * PAGE + (n_valid_last - 1), Poison::data_only);
        const DeviceOut d136 = run_device(q, t136);
        shift_report("max shift when page2 row 8 (t=136, last valid) is mutated",
                     max_abs_diff(clean.attn, d136.attn), 1e-3);
        const Case t137 = poisoned_copy(base, last_page * PAGE + n_valid_last, Poison::data_only);
        const DeviceOut d137 = run_device(q, t137);
        zero_report("max shift when page2 row 9 (t=137, first invalid) is mutated",
                    max_abs_diff(clean.attn, d137.attn));
    }

    // 4) Every page must contribute: page 2 row 0 is token 128.
    {
        const Case t128 = poisoned_copy(base, last_page * PAGE, Poison::data_only);
        const DeviceOut d128 = run_device(q, t128);
        shift_report("max shift when page2 row 0 (t=128) is mutated",
                     max_abs_diff(clean.attn, d128.attn), 1e-3);
    }

    // ---- page-boundary case: 65 tokens, two pages, one token in page 1 ------
    std::printf("\n== controls on %s ==\n", cases[2].name.c_str());
    {
        const Case& cross = cases[2];
        const DeviceOut& cref = dev[2];
        check(cross.n_pages() == 2 && cross.ends[0] == 64 && cross.ends[1] == 65,
              "65 tokens / q_len 2 gives ends [64 65] over two pages");
        const Case t64 = poisoned_copy(cross, 64, Poison::data_only);
        shift_report("max shift when page1 row 0 (t=64) is mutated",
                     max_abs_diff(cref.attn, run_device(q, t64).attn), 1e-3);
        const Case t65 = poisoned_copy(cross, 65, Poison::data_only);
        zero_report("max shift when page1 row 1 (t=65, invalid) is mutated",
                    max_abs_diff(cref.attn, run_device(q, t65).attn));
    }

    // ---- degenerate: empty cache and single token on the device -------------
    std::printf("\n== degenerate lengths on the device ==\n");
    {
        const DeviceOut& empty = dev[6];
        check(empty.attn.size() == static_cast<std::size_t>(HQ) * D,
              "an empty cache still produces one output row per Q head");
        bool zero = true;
        for (float v : empty.attn) {
            if (v != 0.0f) zero = false;
        }
        check(zero, "an empty cache yields exactly zero output rows");

        const DeviceOut& one = dev[5];
        double worst = 0.0;
        for (int pair = 0; pair < HQ; ++pair) {
            for (int d = 0; d < D; ++d) {
                const std::uint8_t byte = cases[5].pages[v_off(0, 0, d >> 1)];
                const int nib = (d & 1) ? static_cast<int>(byte >> 4)
                                        : static_cast<int>(byte & 0x0F);
                const float vs = load_f32(cases[5].pages.data(), m_off(0, 0, Meta::v_scale));
                const float vz = load_f32(cases[5].pages.data(), m_off(0, 0, Meta::v_zero));
                const double v = (static_cast<double>(nib) - static_cast<double>(vz)) *
                                 static_cast<double>(vs);
                worst = std::max(worst,
                                 std::fabs(static_cast<double>(
                                               one.attn[static_cast<std::size_t>(pair) * D +
                                                        static_cast<std::size_t>(d)]) -
                                           v));
            }
        }
        measured("seq1: max |device - decoded V row|", worst);
        derived("seq1: fp16 V dequantisation step on the packed row", 1e-3);
        check(worst < 1e-3, "a one-token cache returns the decoded V row through the device path");
    }

    std::printf("\n%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffered so a device-runtime crash cannot swallow the log.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        return run(argc, argv);
    } catch (const sycl::exception& e) {
        std::printf("SETUP ERROR (sycl): %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::printf("SETUP ERROR: %s\n", e.what());
        return 2;
    }
}
