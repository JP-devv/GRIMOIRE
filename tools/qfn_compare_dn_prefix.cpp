// Localize Flash-Next token-361 DeltaNet/MoE replay in layer 0 or 1.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr size_t rows = 512, position = 361, hidden = 2560, key = 16 * 128, value = 48 * 128, heads = 48;

fs::path capture_path(const fs::path& dir, int layer, int rank, const char* name) {
    return dir / ("l" + std::to_string(layer) + "-p" + std::to_string(position) +
                  "-r" + std::to_string(rank) + "-" + name + ".bin");
}

template <class T = float>
std::vector<T> read(const fs::path& dir, int layer, int rank, const char* name, size_t count) {
    const auto path = capture_path(dir, layer, rank, name);
    std::ifstream file(path, std::ios::binary);
    std::vector<T> values(count);
    if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T)) ||
        file.peek() != std::char_traits<char>::eof() ||
        !std::all_of(values.begin(), values.end(), [](T x) { return std::isfinite(x); }))
        throw std::runtime_error("invalid capture: " + path.string());
    return values;
}

struct Delta {
    double max_abs = 0, rms = 0, relative = 0;
    size_t different = 0, first = std::numeric_limits<size_t>::max();
};

// Compare all rows. A TP rank holds a contiguous segment of each PP row,
// not a contiguous segment of the batched record.
Delta delta(const std::vector<float>& pp, const std::vector<float>& tp, size_t m, size_t width,
            size_t offset, size_t local) {
    if (!local || pp.size() != m * width || tp.size() != m * local || offset + local > width)
        throw std::runtime_error("DeltaNet capture shape mismatch");
    Delta result;
    double squared = 0, ref_squared = 0;
    for (size_t r = 0; r < m; ++r) for (size_t j = 0; j < local; ++j) {
        const double a = pp[r * width + offset + j], b = tp[r * local + j];
        const double d = std::abs(a - b);
        if (d != 0) {
            ++result.different;
            result.first = std::min(result.first, r);
        }
        result.max_abs = std::max(result.max_abs, d);
        squared += d * d;
        ref_squared += a * a;
    }
    result.rms = std::sqrt(squared / tp.size());
    result.relative = result.rms / std::max(1e-12, std::sqrt(ref_squared / tp.size()));
    return result;
}

void print(const char* name, int rank, const Delta& d, size_t m) {
    std::cout << name << " rank=" << rank << " max_abs=" << d.max_abs << " rms=" << d.rms
              << " rel=" << d.relative << " nonzero=" << d.different;
    if (m > 1 && d.different) std::cout << " first_prefix_pos=" << d.first;
    std::cout << '\n';
}

void compare(const fs::path& root, int layer) {
    const auto ppdir = root / "pp4", tpdir = root / "tp4";
    struct Tap { const char* name; size_t m, width; bool sharded; };
    const Tap taps[] = {
        {"entry-span", rows, 4 * hidden, false}, {"dn-input", 1, hidden, false},
        {"dn-q-span", rows, key, true}, {"dn-k-span", rows, key, true},
        {"dn-v-span", rows, value, true}, {"dn-alpha-span", rows, heads, true},
        {"dn-beta-span", rows, heads, true}, {"dn-recurrence", 1, value, true},
        {"dn-projection-input", 1, value, true},
        {"dn-projection-reduced", 1, hidden, false}, {"moe-pre-norm", 1, 4 * hidden, false},
        {"moe-post-norm", 1, 4 * hidden, false}, {"input", 1, hidden, false},
        {"logits", 1, 512, false},
    };
    std::cout << std::setprecision(9);
    const char* first = nullptr;
    int first_rank = -1;
    size_t first_rows = 0;
    Delta first_delta;
    for (const auto& tap : taps) {
        auto pp = read(ppdir, layer, 0, tap.name, tap.m * tap.width);
        const size_t local = tap.sharded ? tap.width / 4 : tap.width;
        const size_t relevant = tap.m == rows && tap.sharded ? position + 1 : tap.m;
        pp.resize(relevant * tap.width);
        for (int rank = 0; rank < 4; ++rank) {
            auto tp = read(tpdir, layer, rank, tap.name, tap.m * local);
            tp.resize(relevant * local);
            if (std::strcmp(tap.name, "entry-span") == 0 &&
                std::memcmp(pp.data(), tp.data(), pp.size() * sizeof(float)) != 0)
                throw std::runtime_error("entry replay was not byte-identical");
            const auto d = delta(pp, tp, relevant, tap.width, tap.sharded ? rank * local : 0, local);
            print(tap.name, rank, d, relevant);
            if (d.different && !first) {
                first = tap.name; first_rank = rank; first_delta = d; first_rows = relevant;
            }
        }
    }
    if (first) { std::cout << "first_differing_tap "; print(first, first_rank, first_delta, first_rows); }
    else std::cout << "first_differing_tap none\n";
    auto rank_partial_sum = read(tpdir, layer, 0, "dn-projection-local", hidden);
    for (int rank = 1; rank < 4; ++rank) {
        const auto partial = read(tpdir, layer, rank, "dn-projection-local", hidden);
        for (size_t i = 0; i < hidden; ++i) rank_partial_sum[i] += partial[i];
    }
    for (int rank = 0; rank < 4; ++rank) {
        const auto actual = read(tpdir, layer, rank, "dn-projection-reduced", hidden);
        print("projection-rank-sum-vs-reduced", rank, delta(rank_partial_sum, actual, 1, hidden, 0, hidden), 1);
    }
    for (const auto& arm : {ppdir, tpdir}) for (int rank = 0; rank < (arm == ppdir ? 1 : 4); ++rank) {
        const auto logits = read(arm, layer, rank, "logits", 512);
        std::vector<int> ids(logits.size());
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = int(i);
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        const auto actual_ids = read<int32_t>(arm, layer, rank, "global", 10);
        if (!std::equal(actual_ids.begin(), actual_ids.end(), ids.begin()))
            throw std::runtime_error("captured router IDs disagree with cutoff logits");
        std::cout << arm.filename().string() << " rank=" << rank << " expert10=" << actual_ids.back()
                  << " expert11=" << ids[10] << " margin=" << (double(logits[ids[9]]) - logits[ids[10]])
                  << " logit492-minus148=" << (double(logits[492]) - logits[148]) << '\n';
    }
}

void self_test() {
    if (capture_path("tp4", 0, 3, "dn-projection-input") != "tp4/l0-p361-r3-dn-projection-input.bin" ||
        capture_path("pp4", 1, 0, "logits") != "pp4/l1-p361-r0-logits.bin")
        throw std::runtime_error("capture filenames must carry the selected layer");
    const std::vector<float> pp{1, 2, 3, 4, 5, 6, 7, 8};
    if (delta(pp, {3, 4, 7, 8}, 2, 4, 2, 2).different != 0)
        throw std::runtime_error("rank slice was not compared within each row");
    const auto d = delta(pp, {3, 4, 7, 9}, 2, 4, 2, 2);
    if (d.different != 1 || d.first != 1 || d.rms != 0.5)
        throw std::runtime_error("prefix divergence was not localized");
    std::cout << "QFN DeltaNet prefix comparator PASS\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") self_test();
        else if (argc == 3 && std::string(argv[2]) == "0") compare(argv[1], 0);
        else if (argc == 2 || (argc == 3 && std::string(argv[2]) == "1")) compare(argv[1], 1);
        else throw std::runtime_error("usage: qfn_compare_dn_prefix DIR [0|1] | --self-test");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
