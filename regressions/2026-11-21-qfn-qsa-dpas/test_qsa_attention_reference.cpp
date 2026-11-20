// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-only contract for the QSA attention recurrence (see README.md).
// Two scalar implementations of the same arithmetic plus the float64 truth,
// so a DPAS mainloop can be judged bitwise before it touches a GPU.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>

namespace {

constexpr int kHeads = 2;      // Q heads in the fixture
constexpr int kKvHeads = 1;
constexpr int kDim = 16;       // head_dim (real: 256; fixture is smaller)
constexpr int kCap = 32;       // seq_cap
constexpr int kRows = 3;

// e4m3 (E2M1 with 3-bit mantissa, bias 7), values only where the fixture uses
// them; the kernel's converter is the authority and this mirrors its arithmetic.
float e4m3(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 0x7;
    const float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (e == 0) return sign * (m / 8.0f) * std::ldexp(1.0f, -6);
    return sign * (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7);
}

struct Fixture {
    std::vector<float> q;                       // [rows][heads][dim]
    std::vector<uint8_t> k, v;                  // [kvh][dim][cap], [kvh][cap][dim]
    std::vector<std::vector<int32_t>> idx;      // per row: selected keys
    float scale = 0.125f;
};

Fixture make_fixture() {
    Fixture f;
    f.q.resize(size_t(kRows) * kHeads * kDim);
    uint32_t s = 12345;
    auto next = [&] { s = s * 1664525u + 1013904223u; return s >> 24; };
    for (auto& v : f.q) v = (float(int(next()) - 128) / 128.0f);
    f.k.resize(size_t(kKvHeads) * kDim * kCap);
    f.v.resize(size_t(kKvHeads) * kCap * kDim);
    for (auto& v : f.k) v = uint8_t(next() & 0x7F);          // finite, positive
    for (auto& v : f.v) v = uint8_t(next() & 0x7F);
    // holes (-1), duplicates, a full run and a partial one
    f.idx = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 11, 11, 13},
             {2, 2, 2, 5, -1, -1, 7, 9, 10, 12, 15, 16, 17, 18, 19, 20},
             {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17}};
    return f;
}

// One head/row of the recurrence, evaluated in list order (thin order).
template <typename Real>
void attention_row(const Fixture& f, int row, int head, std::vector<Real>& out,
                   bool reverse_lists) {
    const int kvh = head / (kHeads / kKvHeads);
    const float* qrow = &f.q[(size_t(row) * kHeads + head) * kDim];
    std::vector<int32_t> list = f.idx[row];
    if (reverse_lists) std::reverse(list.begin(), list.end());
    Real m = -std::numeric_limits<Real>::infinity(), l = Real(0);
    std::vector<Real> acc(kDim, Real(0));
    for (int32_t key : list) {
        Real score = -std::numeric_limits<Real>::infinity();
        if (key >= 0 && key < kCap) {
            Real dot = Real(0);
            for (int d = 0; d < kDim; ++d)
                dot += Real(qrow[d]) * Real(e4m3(f.k[size_t(kvh) * kDim * kCap + size_t(d) * kCap + key]));
            score = dot * Real(f.scale);
        }
        const Real mnew = std::max(m, score);
        const Real corr = std::isinf(m) ? Real(0) : std::exp(m - mnew);
        const Real p = std::isinf(score) ? Real(0) : std::exp(score - mnew);
        l = l * corr + p;
        if (key >= 0 && key < kCap) {
            const uint8_t* vrow = &f.v[size_t(kvh) * kCap * kDim + size_t(key) * kDim];
            for (int d = 0; d < kDim; ++d) acc[d] = acc[d] * corr + p * Real(e4m3(vrow[d]));
        } else {
            for (int d = 0; d < kDim; ++d) acc[d] *= corr;
        }
        m = mnew;
    }
    const Real inv = l > Real(0) ? Real(1) / l : Real(0);
    out.assign(kDim, Real(0));
    for (int d = 0; d < kDim; ++d) out[d] = acc[d] * inv;
}

bool run_case(bool reverse_lists) {
    const Fixture f = make_fixture();
    std::vector<float> f32;
    std::vector<double> f64;
    bool ok = true;
    double worst = 0, denom = 0;
    for (int row = 0; row < kRows; ++row)
        for (int head = 0; head < kHeads; ++head) {
            attention_row<float>(f, row, head, f32, reverse_lists);
            attention_row<double>(f, row, head, f64, reverse_lists);
            for (int d = 0; d < kDim; ++d) {
                const double e = std::fabs(double(f32[d]) - f64[d]);
                worst = std::max(worst, e);
                denom = std::max(denom, std::fabs(f64[d]));
                if (!std::isfinite(f32[d])) ok = false;
            }
        }
    const double rel = denom > 0 ? worst / denom : worst;
    // FP32 reassociation against float64 on this fixture.
    const bool bound_ok = rel <= 2e-6;
    std::printf("QSA reference (%-8s): max|f32-f64| %.3e  rel %.3e  finite=%d  %s\n",
                reverse_lists ? "reversed" : "list", worst, rel, int(ok),
                bound_ok && ok ? "PASS" : "FAIL");
    return bound_ok && ok;
}

// The repack contract: Kp/Vp must be a bijection onto the same (d, s) pairs.
bool repack_maps_agree() {
    // Kp [kvh][d/2][sp][2] : element (d, s) lives at ((d/2)*kCap + s)*2 + (d&1)
    auto kp_index = [](int d, int s) { return size_t(d / 2) * kCap * 2 + size_t(s) * 2 + (d & 1); };
    // Vp [kvh][sp/2][d][2] : element (s, d) lives at ((s/2)*kDim + d)*2 + (s&1)
    auto vp_index = [](int s, int d) { return size_t(s / 2) * kDim * 2 + size_t(d) * 2 + (s & 1); };
    std::vector<int> seen_k(size_t(kDim) * kCap, 0), seen_v(size_t(kDim) * kCap, 0);
    for (int d = 0; d < kDim; ++d)
        for (int s = 0; s < kCap; ++s) {
            const size_t i = kp_index(d, s), j = vp_index(s, d);
            if (i >= seen_k.size() || j >= seen_v.size()) return false;
            seen_k[i]++;
            seen_v[j]++;
        }
    for (int c : seen_k) if (c != 1) return false;
    for (int c : seen_v) if (c != 1) return false;
    std::printf("QSA repack maps: Kp and Vp are bijections over %d elements PASS\n", int(seen_k.size()));
    return true;
}

} // namespace

int main() {
    bool ok = repack_maps_agree();
    ok &= run_case(false);
    ok &= run_case(true);
    std::printf("QSA DPAS contract: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
