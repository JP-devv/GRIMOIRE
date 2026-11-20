// Localize Flash-Next token-361 DeltaNet/MoE replay in layer 0 or 1.
#include "b70/formats.hpp"
#include "b70/safetensors.hpp"

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
template <class A, class B>
Delta delta(const std::vector<A>& pp, const std::vector<B>& tp, size_t m, size_t width,
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

std::vector<double> project_fp64(const std::vector<float>& weights, const std::vector<float>& input,
                                 size_t width, size_t offset) {
    if (!width || weights.empty() || weights.size() % width || input.empty() ||
        offset > width || input.size() > width - offset)
        throw std::runtime_error("projection reference shape mismatch");
    std::vector<double> out(weights.size() / width);
    for (size_t n = 0; n < out.size(); ++n) {
        const float* w = weights.data() + n * width + offset;
        for (size_t k = 0; k < input.size(); ++k) out[n] += double(w[k]) * double(input[k]);
        if (!std::isfinite(out[n])) throw std::runtime_error("nonfinite projection reference");
    }
    return out;
}

void compare_projection(const fs::path& root, int layer, const std::string& shard_path) {
    // The engine's dense BF16 upload copies these tensor bytes directly. TP
    // retains a contiguous column slice of each row, in value-head order.
    b70::SafeTensors shard;
    std::string error;
    if (!shard.open(shard_path, error)) throw std::runtime_error(error);
    const auto* tensor = shard.find("model.language_model.layers." + std::to_string(layer) +
                                    ".linear_attn.out_proj.weight");
    if (!tensor || tensor->dtype != b70::STDtype::BF16 ||
        tensor->shape != std::vector<int64_t>{hidden, value})
        throw std::runtime_error("expected the unscaled BF16 DeltaNet output projection tensor");
    std::vector<float> weights(hidden * value);
    if (!shard.read_f32(*tensor, weights.data(), error)) throw std::runtime_error(error);
    const auto pp = root / "pp4", tp = root / "tp4";
    const auto pp_actual = read(pp, layer, 0, "dn-projection-local", hidden);
    const auto tp_reduced = read(tp, layer, 0, "dn-projection-reduced", hidden);
    for (const bool round_input : {false, true}) {
        const auto input = [&](const fs::path& arm, int rank, size_t width) {
            auto x = read(arm, layer, rank, "dn-projection-input", width);
            if (round_input)
                for (auto& v : x) v = b70::bf16_to_f32(b70::f32_to_bf16(v));
            return x;
        };
        std::cout << "projection_reference operands=" << (round_input ? "bf16-rounded" : "captured-fp32") << '\n';
        const auto full = project_fp64(weights, input(pp, 0, value), value, 0);
        print("pp4-local-vs-fp64", 0, delta(full, pp_actual, 1, hidden, 0, hidden), 1);
        std::vector<double> sum(hidden);
        for (int rank = 0; rank < 4; ++rank) {
            const auto partial = project_fp64(weights, input(tp, rank, value / 4), value, rank * (value / 4));
            const auto actual = read(tp, layer, rank, "dn-projection-local", hidden);
            print("tp4-local-vs-fp64", rank, delta(partial, actual, 1, hidden, 0, hidden), 1);
            for (size_t i = 0; i < hidden; ++i) sum[i] += partial[i];
        }
        print("tp4-reduced-vs-fp64", 0, delta(sum, tp_reduced, 1, hidden, 0, hidden), 1);
        print("tp4-fp64-sum-vs-pp4-fp64", 0, delta(full, sum, 1, hidden, 0, hidden), 1);
    }
}

void self_test() {
    if (capture_path("tp4", 0, 3, "dn-projection-input") != "tp4/l0-p361-r3-dn-projection-input.bin" ||
        capture_path("pp4", 1, 0, "logits") != "pp4/l1-p361-r0-logits.bin")
        throw std::runtime_error("capture filenames must carry the selected layer");
    const std::vector<float> weights{16777216, 1, -16777216, 2, 3, 4};
    if (project_fp64(weights, {1, 1, 1}, 3, 0) != std::vector<double>{1, 9} ||
        project_fp64(weights, {2, 3}, 3, 1) != std::vector<double>{-50331646, 18})
        throw std::runtime_error("projection reference lost FP64 accumulation or sliced the wrong columns");
    const std::vector<float> pp{1, 2, 3, 4, 5, 6, 7, 8};
    if (delta(pp, std::vector<float>{3, 4, 7, 8}, 2, 4, 2, 2).different != 0)
        throw std::runtime_error("rank slice was not compared within each row");
    const auto d = delta(pp, std::vector<float>{3, 4, 7, 9}, 2, 4, 2, 2);
    if (d.different != 1 || d.first != 1 || d.rms != 0.5)
        throw std::runtime_error("prefix divergence was not localized");
    std::cout << "QFN DeltaNet prefix comparator PASS\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") self_test();
        else if (argc >= 2 && argc <= 4 && (argc == 2 || std::string(argv[2]) == "0" || std::string(argv[2]) == "1")) {
            const int layer = argc == 2 ? 1 : argv[2][0] - '0';
            compare(argv[1], layer);
            if (argc == 4) compare_projection(argv[1], layer, argv[3]);
        } else throw std::runtime_error("usage: qfn_compare_dn_prefix DIR [0|1 [PROJECTION_SHARD]] | --self-test");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
