// SPDX-License-Identifier: GPL-3.0-or-later
//
// Host-only port of benchmarks/qfn/compare_moe_row.py.
//
// Standard library only: no SYCL, no GPU, no third-party dependency, so a
// captured resident INT4 MoE row can be re-checked on a host that has no
// oneAPI stack. The checks and the printed report mirror the Python original:
// global-ID ownership, resident remap, the routing cutoff margin, per-expert
// and shared deltas, and the actual FP32 reconstruction of the TP4 partial
// reduction.
// Build (host, no oneAPI):
//   g++ -std=c++17 -O2 -Wall -Wextra -Iinclude tools/qfn_compare_moe_row.cpp
// Usage:
//   qfn_compare_moe_row ROOT [--layer N] [--position N]

#ifndef QFN_COMPARE_MOE_ROW_CPP
#define QFN_COMPARE_MOE_ROW_CPP

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace qfn_moe_row {

namespace fs = std::filesystem;

// Frozen Flash-Next geometry the capture writes: 512 routed experts, top-10
// routing, hidden width 2560, PP4 owner = layer / 12, TP4 world = 4 with 128
// resident experts per rank.
constexpr std::size_t kExperts = 512;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kHidden = 2560;
constexpr int kWorld = 4;
constexpr int kExpertsPerRank = int(kExperts) / kWorld;
constexpr int kLayersPerStage = 12;

constexpr const char* kSummary = "Check captured, resident INT4 MoE rows in native and common-input PP4/TP4.";

// ---------------------------------------------------------------- formatting

// Python's f'{v:.9g}': nine significant digits, shortest of fixed/scientific.
std::string format_g9(double value) {
    char buffer[64];
    const int written = std::snprintf(buffer, sizeof buffer, "%.9g", value);
    if (written <= 0 || std::size_t(written) >= sizeof buffer)
        throw std::runtime_error("format_g9: conversion overflow");
    return std::string(buffer, std::size_t(written));
}

// Python's repr(float): shortest round-tripping digits, fixed notation while
// -4 < decimal-point position <= 16, otherwise scientific with a two-digit
// minimum exponent. Always carries either a '.' or an exponent.
std::string py_float_repr(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::scientific);
    if (result.ec != std::errc()) throw std::runtime_error("py_float_repr: to_chars failed");
    std::string_view text(buffer, std::size_t(result.ptr - buffer));
    bool negative = false;
    if (!text.empty() && text.front() == '-') {
        negative = true;
        text.remove_prefix(1);
    }
    const std::size_t epos = text.find('e');
    if (epos == std::string_view::npos) throw std::runtime_error("py_float_repr: missing exponent");
    std::string digits;
    for (const char c : text.substr(0, epos))
        if (c != '.') digits.push_back(c);
    if (digits.empty()) throw std::runtime_error("py_float_repr: empty mantissa");
    int exponent = 0;
    const std::string_view exponent_text = text.substr(epos + 1);
    const auto parsed = std::from_chars(exponent_text.data(), exponent_text.data() + exponent_text.size(), exponent);
    if (parsed.ec != std::errc()) throw std::runtime_error("py_float_repr: bad exponent");

    const int decimal_point = exponent + 1;  // value == 0.digits * 10^decimal_point
    std::string out;
    if (negative) out.push_back('-');
    if (decimal_point <= -4 || decimal_point > 16) {
        out.push_back(digits.front());
        if (digits.size() > 1) {
            out.push_back('.');
            out.append(digits, 1, std::string::npos);
        }
        out.push_back('e');
        int printed = decimal_point - 1;
        out.push_back(printed < 0 ? '-' : '+');
        if (printed < 0) printed = -printed;
        std::string magnitude = std::to_string(printed);
        if (magnitude.size() < 2) magnitude.insert(magnitude.begin(), '0');
        out += magnitude;
    } else if (decimal_point <= 0) {
        out += "0.";
        out.append(std::size_t(-decimal_point), '0');
        out += digits;
    } else if (std::size_t(decimal_point) >= digits.size()) {
        out += digits;
        out.append(std::size_t(decimal_point) - digits.size(), '0');
        out += ".0";
    } else {
        out.append(digits, 0, std::size_t(decimal_point));
        out.push_back('.');
        out.append(digits, std::size_t(decimal_point), std::string::npos);
    }
    return out;
}

// Python list repr of a sequence of ints: '[1, 128, 257]'.
template <typename Int>
std::string py_int_list(const std::vector<Int>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(values[i]);
    }
    out += "]";
    return out;
}

// Python float-list repr: '[0.125, 0.125]'.
std::string py_float_list(const std::vector<float>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ", ";
        out += py_float_repr(double(values[i]));
    }
    out += "]";
    return out;
}

// NumPy 1-D int array repr, used only by the "invalid global IDs" message.
std::string numpy_int_array(const std::vector<int>& values) {
    std::vector<std::string> cells;
    cells.reserve(values.size());
    std::size_t width = 1;
    for (const int value : values) {
        cells.push_back(std::to_string(value));
        width = std::max(width, cells.back().size());
    }
    std::string out = "[";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (i) out.push_back(' ');
        out.append(width - cells[i].size(), ' ');
        out += cells[i];
    }
    out += "]";
    return out;
}

// ------------------------------------------------------------------- numerics

template <typename To, typename From>
To bit_pun(From value) {
    static_assert(sizeof(To) == sizeof(From), "bit_pun needs equal widths");
    static_assert(std::is_trivially_copyable<To>::value && std::is_trivially_copyable<From>::value,
                  "bit_pun needs trivially copyable types");
    To out;
    std::memcpy(&out, &value, sizeof out);
    return out;
}

std::uint32_t load_le32(const unsigned char* bytes) {
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) | (std::uint32_t(bytes[2]) << 16) |
           (std::uint32_t(bytes[3]) << 24);
}

using Row = std::vector<float>;

struct Delta {
    double rms = 0.0;
    double rel = 0.0;
    double max_abs = 0.0;

    std::string str() const {
        return "rms=" + format_g9(rms) + " rel=" + format_g9(rel) + " max_abs=" + format_g9(max_abs);
    }
};

// delta(a, b): RMS of (float64(a) - b), relative to a's own RMS, plus max |d|.
Delta delta(const Row& a, const Row& b) {
    if (a.empty() || a.size() != b.size()) throw std::runtime_error("delta: operand width mismatch");
    Delta out;
    double squared = 0, ref_squared = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double lhs = a[i], diff = lhs - double(b[i]);
        squared += diff * diff;
        ref_squared += lhs * lhs;
        out.max_abs = std::max(out.max_abs, std::fabs(diff));
    }
    out.rms = std::sqrt(squared / a.size());
    out.rel = out.rms / std::max(1e-12, std::sqrt(ref_squared / a.size()));
    return out;
}

std::string delta_str(const Row& a, const Row& b) {
    return delta(a, b).str();
}

// Python's fold(): copy the first array, then add the rest in order. The
// accumulation is float32, exactly as NumPy does it.
Row fold(const std::vector<const Row*>& parts) {
    if (parts.empty()) throw std::runtime_error("fold: no operands");
    Row result = *parts.front();
    for (std::size_t p = 1; p < parts.size(); ++p) {
        const Row& addend = *parts[p];
        if (addend.size() != result.size())
            throw std::runtime_error("fold: operand width " + std::to_string(addend.size()) + " vs " +
                                     std::to_string(result.size()));
        for (std::size_t i = 0; i < result.size(); ++i) result[i] += addend[i];
    }
    return result;
}

// Elementwise float32 sum, matching row['routed'] + row['shared'].
Row add_rows(const Row& a, const Row& b) {
    if (a.size() != b.size()) throw std::runtime_error("add_rows: width mismatch");
    Row result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

bool same_bytes(const Row& a, const Row& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool same_bytes(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(int)) == 0);
}

// ---------------------------------------------------------------------- input

std::vector<unsigned char> read_bytes(const fs::path& path, std::size_t expected) {
    std::error_code ec;
    const std::uintmax_t size = fs::file_size(path, ec);
    if (ec) throw std::runtime_error(path.string() + ": " + ec.message());
    if (size != expected) throw std::runtime_error(path.string() + ": wrong byte size");
    std::vector<unsigned char> bytes(expected);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path.string() + ": cannot open for reading");
    if (expected && !in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(expected)))
        throw std::runtime_error(path.string() + ": truncated read");
    return bytes;
}

std::vector<int> read_i32(const fs::path& path, std::size_t count) {
    const std::vector<unsigned char> bytes = read_bytes(path, count * 4);
    std::vector<int> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = int(bit_pun<std::int32_t>(load_le32(bytes.data() + 4 * i)));
    return out;
}

std::vector<float> read_f32(const fs::path& path, std::size_t count) {
    const std::vector<unsigned char> bytes = read_bytes(path, count * 4);
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = bit_pun<float>(load_le32(bytes.data() + 4 * i));
        if (!std::isfinite(out[i])) throw std::runtime_error(path.string() + ": non-finite data");
    }
    return out;
}

// ------------------------------------------------------------------- the rows

struct MoERow {
    std::vector<int> ids;         // int32 global expert IDs, capture order
    std::vector<int> local_ids;   // int32 resident IDs (-1 when not resident here)
    std::vector<float> weights;   // FP32 gate weights
    std::vector<float> logits;    // FP32 gate logits over all 512 experts
    std::map<int, Row> contributions;  // global expert ID -> weighted down output
    Row native_input;             // FP32 row before the routing probe replaced it
    Row input;                    // FP32 row actually entering the router
    Row routed;
    Row shared;
    Row partial;                  // routed + this rank's share of shared
    Row reduced;                  // after the TP4 reduction
};

fs::path record_path(const fs::path& directory, int layer, int pos, int rank, const std::string& name) {
    return directory / ("l" + std::to_string(layer) + "-p" + std::to_string(pos) + "-r" + std::to_string(rank) + "-" +
                        name + ".bin");
}

MoERow load_row(const fs::path& directory, int layer, int pos, int rank, bool tensor_parallel) {
    const auto get_i32 = [&](const std::string& name, std::size_t count) {
        return read_i32(record_path(directory, layer, pos, rank, name), count);
    };
    const auto get_f32 = [&](const std::string& name, std::size_t count) {
        return read_f32(record_path(directory, layer, pos, rank, name), count);
    };

    MoERow row;
    row.ids = get_i32("global", kTopK);
    row.local_ids = get_i32("local", kTopK);
    row.logits = get_f32("logits", kExperts);
    row.weights = get_f32("weights", kTopK);

    {
        std::vector<int> sorted_ids = row.ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        const bool duplicate = std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end();
        bool out_of_range = false;
        for (const int id : row.ids) out_of_range = out_of_range || id < 0 || id >= int(kExperts);
        if (duplicate || out_of_range)
            throw std::runtime_error(directory.string() + ": invalid global IDs " + numpy_int_array(row.ids));
    }

    // GPU top-k resolves exact ties by the lowest global expert ID, which is a
    // stable sort over the logits in descending order.
    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&row](int a, int b) { return row.logits[std::size_t(a)] > row.logits[std::size_t(b)]; });
    for (std::size_t i = 0; i < kTopK; ++i)
        if (order[i] != row.ids[i])
            throw std::runtime_error(directory.string() + " rank " + std::to_string(rank) +
                                     ": captured logit order disagrees with GPU top-k");

    std::vector<int> expected(kTopK);
    for (std::size_t i = 0; i < kTopK; ++i) {
        const int global = row.ids[i];
        expected[i] = global;
        if (tensor_parallel) {
            const int begin = rank * kExpertsPerRank;
            expected[i] = (begin <= global && global < begin + kExpertsPerRank) ? global - begin : -1;
        }
    }

    std::cout << directory.filename().string() << " rank=" << rank << " top_k=" << py_int_list(row.ids)
              << " weights=" << py_float_list(row.weights) << '\n';

    const std::size_t tenth = kTopK - 1, eleventh = kTopK;
    const int e10 = order[tenth], e11 = order[eleventh];
    std::cout << "  cutoff: " << e10 << ':' << format_g9(double(row.logits[std::size_t(e10)])) << " vs " << e11 << ':'
              << format_g9(double(row.logits[std::size_t(e11)]))
              << "; margin=" << format_g9(double(row.logits[std::size_t(e10)]) - double(row.logits[std::size_t(e11)]))
              << "; local_ids=" << py_int_list(row.local_ids) << " remap=" << (expected == row.local_ids ? "OK" : "MISMATCH")
              << '\n';

    for (std::size_t i = 0; i < kTopK; ++i)
        if (row.local_ids[i] >= 0)
            row.contributions.emplace(row.ids[i], get_f32("expert-" + std::to_string(row.ids[i]), kHidden));
    row.native_input = get_f32("native-input", kHidden);
    row.input = get_f32("input", kHidden);
    row.routed = get_f32("routed", kHidden);
    row.shared = get_f32("shared", kHidden);
    row.partial = get_f32("partial", kHidden);
    row.reduced = get_f32("reduced", kHidden);

    // The captured routed row must be the ordered sum of the resident slots,
    // with a non-resident slot contributing an explicit zero.
    std::vector<Row> ordered_slots;
    ordered_slots.reserve(kTopK);
    for (const int global : row.ids) {
        const auto slot = row.contributions.find(global);
        ordered_slots.push_back(slot == row.contributions.end() ? Row(kHidden, 0.0f) : slot->second);
    }
    std::vector<const Row*> ordered_ptrs;
    ordered_ptrs.reserve(ordered_slots.size());
    for (const Row& slot : ordered_slots) ordered_ptrs.push_back(&slot);

    std::cout << "  routed vs captured slots: " << delta_str(row.routed, fold(ordered_ptrs))
              << "; partial vs routed+shared: " << delta_str(row.partial, add_rows(row.routed, row.shared)) << '\n';
    return row;
}

// ------------------------------------------------------------------- the arms

struct Arm {
    MoERow pp;                 // PP4 owner of this layer
    std::vector<MoERow> tp;    // all four TP4 ranks
};

Arm compare_arm(const fs::path& directory, int layer, int pos) {
    Arm arm;
    arm.pp = load_row(directory / "pp4", layer, pos, layer / kLayersPerStage, false);
    arm.tp.reserve(kWorld);
    for (int rank = 0; rank < kWorld; ++rank) arm.tp.push_back(load_row(directory / "tp4", layer, pos, rank, true));

    for (int rank = 0; rank < kWorld; ++rank) {
        const MoERow& t = arm.tp[std::size_t(rank)];
        std::cout << "PP4 vs TP4 rank=" << rank << ": input " << delta_str(arm.pp.input, t.input) << "; logits "
                  << delta_str(arm.pp.logits, t.logits) << "; output " << delta_str(arm.pp.reduced, t.reduced) << '\n';
    }

    {
        std::vector<const Row*> partials;
        partials.reserve(arm.tp.size());
        for (const MoERow& t : arm.tp) partials.push_back(&t.partial);
        const Row summed = fold(partials);
        for (int rank = 0; rank < kWorld; ++rank)
            std::cout << "TP4 sum vs rank=" << rank << " reduced: " << delta_str(summed, arm.tp[std::size_t(rank)].reduced)
                      << '\n';
    }

    {
        std::vector<const Row*> shared;
        shared.reserve(arm.tp.size());
        for (const MoERow& t : arm.tp) shared.push_back(&t.shared);
        std::cout << "PP4 vs sum of TP4 shared: " << delta_str(arm.pp.shared, fold(shared)) << '\n';
    }

    std::map<int, std::pair<int, const Row*>> merged;
    for (int rank = 0; rank < kWorld; ++rank)
        for (const auto& entry : arm.tp[std::size_t(rank)].contributions) {
            if (merged.find(entry.first) != merged.end())
                throw std::runtime_error("duplicate TP contribution for expert " + std::to_string(entry.first));
            merged.emplace(entry.first, std::make_pair(rank, &entry.second));
        }

    std::vector<int> only_pp, only_tp;
    for (const auto& entry : arm.pp.contributions)
        if (merged.find(entry.first) == merged.end()) only_pp.push_back(entry.first);
    for (const auto& entry : merged)
        if (arm.pp.contributions.find(entry.first) == arm.pp.contributions.end()) only_tp.push_back(entry.first);
    std::cout << "only PP4: " << py_int_list(only_pp) << "; only TP4: " << py_int_list(only_tp) << '\n';

    for (const auto& entry : merged) {
        const auto pp_contribution = arm.pp.contributions.find(entry.first);
        if (pp_contribution == arm.pp.contributions.end()) continue;
        std::cout << "global expert=" << entry.first << " owner=" << entry.second.first << " weighted PP4 vs TP4: "
                  << delta_str(pp_contribution->second, *entry.second.second) << '\n';
    }
    return arm;
}

bool top_k_equal(const Arm& arm) {
    for (const MoERow& t : arm.tp)
        if (!same_bytes(t.ids, arm.pp.ids)) return false;
    return true;
}

// Full report: the native capture pair, then the common-input pair, whose PP4
// input must be the exact native PP4 input bytes.
int run(const fs::path& root, int layer, int pos) {
    std::cout << "=== Native inputs ===" << '\n';
    std::cout.flush();
    const Arm native = compare_arm(root / "native", layer, pos);

    std::cout << "=== Same captured PP4 input ===" << '\n';
    std::cout.flush();
    const Arm common = compare_arm(root / "common", layer, pos);

    if (!same_bytes(common.pp.input, native.pp.input))
        throw std::runtime_error("common PP4 did not receive the exact PP4 input bytes");
    for (int rank = 0; rank < kWorld; ++rank)
        if (!same_bytes(common.tp[std::size_t(rank)].input, native.pp.input))
            throw std::runtime_error("common TP4-" + std::to_string(rank) +
                                     " did not receive the exact PP4 input bytes");
    if (!same_bytes(common.pp.logits, native.pp.logits) || !same_bytes(common.pp.ids, native.pp.ids) ||
        !same_bytes(common.pp.weights, native.pp.weights) || !same_bytes(common.pp.reduced, native.pp.reduced))
        throw std::runtime_error("common-input PP4 failed to reproduce its native reference");

    std::cout << "BOUNDARY native_topk_equal=" << (top_k_equal(native) ? "True" : "False")
              << " common_topk_equal=" << (top_k_equal(common) ? "True" : "False") << '\n';
    return 0;
}

// ---------------------------------------------------------------- command line

std::string program_name(const char* argv0) {
    if (!argv0) return "qfn_compare_moe_row";
    const std::string name = fs::path(argv0).filename().string();
    return name.empty() ? std::string("qfn_compare_moe_row") : name;
}

int usage_error(const std::string& program, const std::string& message) {
    std::cerr << "usage: " << program << " [-h] [--layer LAYER] [--position POSITION] root" << '\n'
              << program << ": error: " << message << '\n';
    return 2;
}

int cli(int argc, char** argv) {
    const std::string program = program_name(argc > 0 ? argv[0] : nullptr);
    int layer = 1, position = 361;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto read_value = [&](const std::string& option, std::string& value) {
            if (i + 1 >= argc) return usage_error(program, "argument " + option + ": expected one argument");
            value = argv[++i];
            return 0;
        };
        if (argument == "-h" || argument == "--help") {
            std::cout << kSummary << "\n\n"
                      << "positional arguments:\n  root\n\n"
                      << "options:\n  -h, --help           show this help message and exit\n"
                      << "  --layer LAYER\n  --position POSITION\n\n"
                      << "usage: " << program << " [-h] [--layer LAYER] [--position POSITION] root" << '\n';
            return 0;
        }
        std::string name = argument, value;
        bool inline_value = false;
        const std::size_t equals = argument.find('=');
        if (argument.compare(0, 2, "--") == 0 && equals != std::string::npos) {
            name = argument.substr(0, equals);
            value = argument.substr(equals + 1);
            inline_value = true;
        }
        if (name == "--layer" || name == "--position") {
            if (!inline_value) {
                const int status = read_value(name, value);
                if (status != 0) return status;
            }
            int parsed = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc() || result.ptr != value.data() + value.size())
                return usage_error(program, "argument " + name + ": invalid int value: '" + value + "'");
            (name == "--layer" ? layer : position) = parsed;
            continue;
        }
        if (argument.size() > 1 && argument.front() == '-')
            return usage_error(program, "unrecognized arguments: " + argument);
        positional.push_back(argument);
    }

    if (positional.empty()) return usage_error(program, "the following arguments are required: root");
    if (positional.size() > 1)
        return usage_error(program, "unrecognized arguments: " + positional[1]);

    try {
        return run(positional.front(), layer, position);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

}  // namespace qfn_moe_row

#ifndef QFN_COMPARE_MOE_ROW_NO_MAIN
int main(int argc, char** argv) {
    return qfn_moe_row::cli(argc, argv);
}
#endif

#endif  // QFN_COMPARE_MOE_ROW_CPP
