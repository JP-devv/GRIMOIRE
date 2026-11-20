// SPDX-License-Identifier: GPL-3.0-or-later
//
// Host-only C++17 port of benchmarks/qfn/compare_prefill_activations.py.
//
// Reads the PP4 and TP4 activation dumps written by GRIMOIRE_QFN_ACT_DIR and
// compares every TP4 rank's rows against the row produced by the owning PP4
// stage. The dump is streamed one frozen chunk at a time: for a chunk the eight
// rank files (pp4x4, tp4x4) are opened, every record header is validated and
// indexed into per-(layer, tap) spans, and each span is then read back with a
// seek instead of mapping a whole file. Two row buffers (reference, actual) are
// reused across layers.
//
// Output is identical to the Python scanner, including the "checked [a,b)"
// progress line per chunk and the first_nonzero / first_material ordering.
//
// Standard C++ only: no Python, no NumPy, no SYCL, no GPU.
//
// Build (host only):
//   g++ -std=c++17 -O2 -Wall -Wextra -Iinclude tools/qfn_compare_prefill_activations.cpp
//       -o bin/qfn_compare_prefill_activations
//
// Self-test:
//   regressions/2026-11-19-qfn-prefill-activations/test_qfn_prefill_activations.cpp

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace qfn_act {

// ---------------------------------------------------------------- geometry --

inline constexpr std::uint32_t magic = 0x31414651u; // "QFA1" little endian
inline constexpr std::uint32_t header_size = 7 * 4;
inline constexpr std::uint32_t layer_count = 48;
inline constexpr std::uint32_t pp_world = 4;
inline constexpr std::uint32_t tp_world = 4;
inline constexpr std::uint32_t layers_per_stage = layer_count / pp_world;
inline constexpr std::uint32_t max_width = 10240;
inline constexpr std::uint32_t compare_block = 32;

inline constexpr std::array<const char*, 7> tap_names = {
    "embed", "entry", "ple_table", "ple_residual", "mixer", "moe", "exit"};
// -1 marks a tap with no frozen width.
inline constexpr std::array<int, 7> expected_width = {10240, 10240, -1, 10240, 2560, 2560, 10240};

struct Chunk {
    std::uint32_t start;
    std::uint32_t rows;
};

inline const std::vector<Chunk>& frozen_chunks() {
    static const std::vector<Chunk> chunks = {
        {0, 512}, {512, 512}, {1024, 512}, {1536, 512}, {2048, 512}, {2560, 146}};
    return chunks;
}

struct Header {
    std::uint32_t magic;
    std::uint32_t tap;
    std::uint32_t layer;
    std::uint32_t rank;
    std::uint32_t pos;
    std::uint32_t count;
    std::uint32_t width;
};

struct Span {
    std::uint32_t pos;
    std::uint32_t count;
    std::uint32_t width;
    std::uint64_t offset;
};

using Key = std::pair<std::uint32_t, std::uint32_t>; // (layer, tap)
using SpanMap = std::map<Key, std::vector<Span>>;
using SummaryKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>; // layer, tap, tp rank

// ------------------------------------------------------- little-endian I/O --

inline constexpr bool host_little_endian =
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#else
    false; // conservative: decode byte-wise
#endif

std::uint32_t load_u32le(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}

void store_u32le(unsigned char* p, std::uint32_t v) {
    p[0] = static_cast<unsigned char>(v);
    p[1] = static_cast<unsigned char>(v >> 8);
    p[2] = static_cast<unsigned char>(v >> 16);
    p[3] = static_cast<unsigned char>(v >> 24);
}

float load_f32le(const unsigned char* p) {
    static_assert(std::numeric_limits<float>::is_iec559, "activation payload is IEEE-754 binary32");
    const std::uint32_t bits = load_u32le(p);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

void store_f32le(unsigned char* p, float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    store_u32le(p, bits);
}

// ------------------------------------------------------------- formatting --

std::string format_g(double v, int significant) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", significant, v);
    return buf;
}

std::string format_g6(double v) { return format_g(v, 6); }
std::string format_g8(double v) { return format_g(v, 8); }

std::string key_text(const Key& k) {
    return "(" + std::to_string(k.first) + ", " + std::to_string(k.second) + ")";
}

std::string header_text(const Header& h) {
    return "(" + std::to_string(h.magic) + ", " + std::to_string(h.tap) + ", " +
           std::to_string(h.layer) + ", " + std::to_string(h.rank) + ", " + std::to_string(h.pos) +
           ", " + std::to_string(h.count) + ", " + std::to_string(h.width) + ")";
}

// ------------------------------------------------------------------ errors --

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------- file access --

std::uint64_t file_size(const std::string& name) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(name, ec);
    if (ec) throw std::runtime_error(name + ": " + ec.message());
    return size;
}

std::string path_text(const std::filesystem::path& p) { return p.string(); }

std::string record_path(const std::filesystem::path& root, const char* tag, std::uint32_t rank,
                        std::uint32_t start) {
    return path_text(root / tag / ("r" + std::to_string(rank) + "-p" + std::to_string(start) + ".bin"));
}

std::ifstream open_input(const std::string& name) {
    std::ifstream in(name, std::ios::binary);
    if (!in) throw std::runtime_error(name + ": cannot open");
    return in;
}

// Reads one 28-byte header. Returns nullopt at a clean end of file, matching the
// Python `while data := file.read(HEADER.size)` loop.
std::optional<Header> read_header(std::istream& in, const std::string& name) {
    unsigned char raw[header_size];
    in.read(reinterpret_cast<char*>(raw), std::streamsize(header_size));
    const std::streamsize got = in.gcount();
    if (got == 0 && in.eof()) return std::nullopt;
    if (got != std::streamsize(header_size)) throw std::runtime_error(name + ": truncated header");
    Header h;
    h.magic = load_u32le(raw + 0);
    h.tap = load_u32le(raw + 4);
    h.layer = load_u32le(raw + 8);
    h.rank = load_u32le(raw + 12);
    h.pos = load_u32le(raw + 16);
    h.count = load_u32le(raw + 20);
    h.width = load_u32le(raw + 24);
    return h;
}

// Validates every record of one rank file for one frozen chunk and returns the
// (layer, tap) -> spans index. Payload bytes are skipped, not read.
SpanMap index_rank(std::istream& in, const std::string& name, std::uint32_t rank,
                   std::uint32_t first, std::uint32_t rows) {
    const std::uint64_t size = file_size(name);
    SpanMap spans;
    for (;;) {
        const std::optional<Header> next = read_header(in, name);
        if (!next) break;
        const Header& h = *next;
        const auto offset = static_cast<std::uint64_t>(in.tellg());
        const std::uint64_t payload = std::uint64_t(h.count) * h.width * 4;
        const bool bad = h.magic != magic || h.tap >= tap_names.size() || h.layer >= layer_count ||
                         h.rank != rank || !(h.count > 0 && h.count <= rows) ||
                         !(h.width > 0 && h.width <= max_width) ||
                         !(first <= h.pos && std::uint64_t(h.pos) + h.count <= std::uint64_t(first) + rows) ||
                         offset + payload > size;
        if (bad) throw std::runtime_error(name + ": invalid record " + header_text(h));
        spans[{h.layer, h.tap}].push_back(Span{h.pos, h.count, h.width, offset});
        in.seekg(std::streamoff(payload), std::ios::cur);
        if (!in) throw std::runtime_error(name + ": seek past record payload failed");
    }
    for (const auto& entry : spans) {
        const std::vector<Span>& records = entry.second;
        std::uint64_t cursor = first;
        const std::uint32_t width = records.front().width;
        for (const Span& s : records) {
            if (s.pos != cursor || s.width != width)
                throw std::runtime_error(name + ": overlapping/gapped rows or width drift at " +
                                         key_text(entry.first));
            cursor += s.count;
        }
        if (cursor != std::uint64_t(first) + rows)
            throw std::runtime_error(name + ": missing tail at " + key_text(entry.first));
    }
    return spans;
}

void bytes_to_floats(const std::vector<unsigned char>& bytes, float* dest) {
    if constexpr (host_little_endian) {
        static_assert(std::numeric_limits<float>::is_iec559, "activation payload is IEEE-754 binary32");
        std::memcpy(dest, bytes.data(), bytes.size());
    } else {
        const std::size_t n = bytes.size() / 4;
        for (std::size_t i = 0; i < n; ++i) dest[i] = load_f32le(bytes.data() + i * 4);
    }
}

// Re-reads the spans of one key in file order into a rows x width FP32 buffer.
// The index already proved the spans tile [first, first+rows) exactly.
void load_spans(std::istream& in, const std::vector<Span>& spans, std::uint32_t rows,
                std::uint32_t width, float* target, std::vector<unsigned char>& bytes,
                const std::string& name) {
    std::size_t filled = 0;
    for (const Span& s : spans) {
        in.clear();
        in.seekg(std::streamoff(s.offset));
        if (!in) throw std::runtime_error(name + ": seek to record payload failed");
        const std::size_t want = std::size_t(s.count) * s.width * sizeof(float);
        bytes.resize(want);
        in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(want));
        if (std::size_t(in.gcount()) != want) throw std::runtime_error(name + ": truncated payload");
        bytes_to_floats(bytes, target + filled);
        filled += std::size_t(s.count) * s.width;
    }
    if (filled != std::size_t(rows) * width)
        throw std::runtime_error(name + ": span coverage mismatch");
}

// --------------------------------------------------------------- comparison --

// Row-wise RMS delta / relative RMS delta over one rows x width block group.
// A block of up to 32 rows is validated for finiteness before any of its rows
// are reported, exactly like the NumPy path.
template <class Sink>
void compare_rows(const float* reference, const float* actual, std::uint32_t rows,
                  std::uint32_t width, double abs_floor, double rel_floor, Sink&& sink) {
    for (std::uint32_t base = 0; base < rows; base += compare_block) {
        const std::uint32_t block = std::min(compare_block, rows - base);
        for (std::uint32_t r = 0; r < block; ++r) {
            const float* a = reference + std::size_t(base + r) * width;
            const float* b = actual + std::size_t(base + r) * width;
            for (std::uint32_t c = 0; c < width; ++c)
                if (!std::isfinite(a[c]) || !std::isfinite(b[c]))
                    throw std::runtime_error("nonfinite activation");
        }
        for (std::uint32_t r = 0; r < block; ++r) {
            const float* a = reference + std::size_t(base + r) * width;
            const float* b = actual + std::size_t(base + r) * width;
            double square_diff = 0.0, square_ref = 0.0;
            for (std::uint32_t c = 0; c < width; ++c) {
                const double d = double(a[c]) - double(b[c]);
                const double v = double(a[c]);
                square_diff += d * d;
                square_ref += v * v;
            }
            const double diff_rms = std::sqrt(square_diff / double(width));
            const double ref_rms = std::sqrt(square_ref / double(width));
            const double relative = diff_rms / std::max(ref_rms, 1e-12);
            const bool material = diff_rms >= abs_floor && relative >= rel_floor;
            sink(base + r, diff_rms, relative, material);
        }
    }
}

// --------------------------------------------------------------------- run --

struct Stats {
    std::uint64_t rows = 0;
    std::uint64_t material_rows = 0;
    double max_rms = 0.0;
    double max_rel_rms = 0.0;
};

struct Hit {
    std::uint32_t layer = 0;
    std::uint32_t tap = 0;
    std::uint32_t pos = 0;
    std::uint32_t rank = 0;
    double rms = 0.0;
    double rel_rms = 0.0;
};

bool hit_less(const Hit& a, const Hit& b) {
    return std::make_tuple(a.layer, a.tap, a.pos, a.rank, a.rms, a.rel_rms) <
           std::make_tuple(b.layer, b.tap, b.pos, b.rank, b.rms, b.rel_rms);
}

struct Result {
    std::map<SummaryKey, Stats> summary;
    std::optional<Hit> first_nonzero;
    std::optional<Hit> first_material;
    std::uint64_t compared_rows = 0;
};

std::vector<Key> select_keys(std::optional<std::uint32_t> selected) {
    std::vector<Key> keys;
    for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
        if (selected && *selected != layer) continue;
        for (std::uint32_t tap = 0; tap < tap_names.size(); ++tap) {
            const bool wanted = tap == 1 || tap == 6 || (tap == 0 && layer == 0) ||
                                (selected && (tap == 4 || tap == 5 ||
                                              (layer == 1 && (tap == 2 || tap == 3)))); // frozen ple_layer_ids=[2] is one-based
            if (wanted) keys.push_back(Key{layer, tap});
        }
    }
    return keys;
}

Result run(const std::filesystem::path& root, std::optional<std::uint32_t> selected,
           double abs_floor, double rel_floor, std::ostream& out,
           const std::vector<Chunk>& chunks = frozen_chunks()) {
    const std::vector<Key> keys = select_keys(selected);
    Result result;

    std::vector<float> reference, actual;
    std::vector<unsigned char> bytes;

    for (const Chunk& chunk : chunks) {
        std::array<std::string, pp_world> pp_names, tp_names;
        std::array<std::ifstream, pp_world> pp_files;
        std::array<std::ifstream, tp_world> tp_files;
        std::array<SpanMap, pp_world> pp_index;
        std::array<SpanMap, tp_world> tp_index;
        for (std::uint32_t rank = 0; rank < pp_world; ++rank) {
            pp_names[rank] = record_path(root, "pp4", rank, chunk.start);
            pp_files[rank] = open_input(pp_names[rank]);
        }
        for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
            tp_names[rank] = record_path(root, "tp4", rank, chunk.start);
            tp_files[rank] = open_input(tp_names[rank]);
        }
        for (std::uint32_t rank = 0; rank < pp_world; ++rank)
            pp_index[rank] = index_rank(pp_files[rank], pp_names[rank], rank, chunk.start, chunk.rows);
        for (std::uint32_t rank = 0; rank < tp_world; ++rank)
            tp_index[rank] = index_rank(tp_files[rank], tp_names[rank], rank, chunk.start, chunk.rows);

        for (const Key& key : keys) {
            const std::uint32_t owner = key.first / layers_per_stage;
            const auto pp_it = pp_index[owner].find(key);
            std::array<const std::vector<Span>*, tp_world> tp{};
            bool missing = pp_it == pp_index[owner].end();
            for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
                const auto it = tp_index[rank].find(key);
                if (it == tp_index[rank].end()) {
                    missing = true;
                } else {
                    tp[rank] = &it->second;
                }
            }
            if (missing)
                throw std::runtime_error("missing " + key_text(key) + " at chunk " +
                                         std::to_string(chunk.start) + ": PP4 owner " +
                                         std::to_string(owner) + " or one of TP4");

            const std::uint32_t width = pp_it->second.front().width;
            const int want = expected_width[key.second];
            if ((want >= 0 && std::uint32_t(want) != width) ||
                std::any_of(tp.begin(), tp.end(),
                            [width](const std::vector<Span>* s) { return s->front().width != width; }))
                throw std::runtime_error("width mismatch for " + key_text(key) + " at " +
                                         std::to_string(chunk.start));

            const std::size_t cells = std::size_t(chunk.rows) * width;
            reference.resize(cells);
            actual.resize(cells);
            load_spans(pp_files[owner], pp_it->second, chunk.rows, width, reference.data(), bytes,
                       pp_names[owner]);

            for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
                load_spans(tp_files[rank], *tp[rank], chunk.rows, width, actual.data(), bytes,
                           tp_names[rank]);
                compare_rows(
                    reference.data(), actual.data(), chunk.rows, width, abs_floor, rel_floor,
                    [&](std::uint32_t row, double rms, double rel, bool material) {
                        const Hit hit{key.first, key.second, chunk.start + row, rank, rms, rel};
                        if (rms != 0.0 && (!result.first_nonzero || hit_less(hit, *result.first_nonzero)))
                            result.first_nonzero = hit;
                        if (material && (!result.first_material || hit_less(hit, *result.first_material)))
                            result.first_material = hit;
                        Stats& stats = result.summary[SummaryKey{key.first, key.second, rank}];
                        stats.rows += 1;
                        stats.max_rms = std::max(stats.max_rms, rms);
                        stats.max_rel_rms = std::max(stats.max_rel_rms, rel);
                        if (material) stats.material_rows += 1;
                        result.compared_rows += 1;
                    });
            }
        }

        out << "checked [" << chunk.start << ',' << chunk.start + chunk.rows
            << ") vs all TP4 ranks\n";
        out.flush();
    }

    for (const auto& entry : result.summary) {
        const std::uint32_t layer = std::get<0>(entry.first);
        const std::uint32_t tap = std::get<1>(entry.first);
        const std::uint32_t rank = std::get<2>(entry.first);
        const Stats& stats = entry.second;
        out << "layer=" << layer << " tap=" << tap_names[tap] << " tp_rank=" << rank
            << " rows=" << stats.rows << " max_rms=" << format_g8(stats.max_rms)
            << " max_rel_rms=" << format_g8(stats.max_rel_rms)
            << " material_rows=" << stats.material_rows << '\n';
    }
    const std::array<std::pair<const char*, const std::optional<Hit>*>, 2> hits = {
        std::make_pair("first_nonzero", &result.first_nonzero),
        std::make_pair("first_material", &result.first_material)};
    for (const auto& entry : hits) {
        out << entry.first;
        if (!*entry.second) {
            out << " none\n";
            continue;
        }
        const Hit& hit = **entry.second;
        out << " layer=" << hit.layer << " tap=" << tap_names[hit.tap] << " pos=" << hit.pos
            << " tp_rank=" << hit.rank << " rms=" << format_g8(hit.rms)
            << " rel_rms=" << format_g8(hit.rel_rms) << '\n';
    }
    out.flush();
    return result;
}

// ------------------------------------------------------------------ cli -----

struct Options {
    std::filesystem::path root;
    std::optional<std::uint32_t> layer;
    double abs_rms = 1e-4;
    double rel_rms = 0.01;
};

double parse_double(const std::string& text, const std::string& flag) {
    const char* begin = text.c_str();
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin || *end != '\0' || text.empty())
        throw UsageError(flag + ": expected a number, got '" + text + "'");
    return value;
}

std::uint32_t parse_layer(const std::string& text, const std::string& flag) {
    if (text.empty()) throw UsageError(flag + ": expected a layer, got ''");
    const char* begin = text.c_str();
    if (*begin == '+') ++begin;
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(begin, text.c_str() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.c_str() + text.size() || value >= layer_count)
        throw UsageError(flag + ": expected an integer in [0, " + std::to_string(layer_count - 1) +
                         "], got '" + text + "'");
    return value;
}

// Returns nullopt when help was requested.
std::optional<Options> parse_options(const std::vector<std::string>& args) {
    Options options;
    bool have_root = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "-h" || arg == "--help") return std::nullopt;
        if (!arg.empty() && arg[0] == '-' && arg != "-") {
            std::string flag = arg, value;
            const std::size_t eq = arg.find('=');
            const bool inline_value = eq != std::string::npos;
            if (inline_value) {
                flag = arg.substr(0, eq);
                value = arg.substr(eq + 1);
            }
            const bool known = flag == "--layer" || flag == "--abs-rms" || flag == "--rel-rms";
            if (!known) throw UsageError("unrecognised argument '" + arg + "'");
            if (!inline_value) {
                if (i + 1 >= args.size()) throw UsageError(flag + ": expected a value");
                value = args[++i];
            }
            if (flag == "--layer") {
                options.layer = parse_layer(value, flag);
            } else if (flag == "--abs-rms") {
                options.abs_rms = parse_double(value, flag);
            } else {
                options.rel_rms = parse_double(value, flag);
            }
            continue;
        }
        if (have_root) throw UsageError("unrecognised argument '" + arg + "'");
        options.root = arg;
        have_root = true;
    }
    if (!have_root) throw UsageError("the following arguments are required: root");
    return options;
}

inline constexpr const char* usage_text =
    "usage: qfn_compare_prefill_activations [-h] [--layer {0..47}] [--abs-rms ABS_RMS]\n"
    "                                      [--rel-rms REL_RMS] root\n";

std::string describe(const Options& options) {
    return "material iff RMS delta >= " + format_g6(options.abs_rms) +
           " and relative RMS delta >= " + format_g6(options.rel_rms) +
           "; ordered by layer, tap, absolute token, TP rank";
}

} // namespace qfn_act

#ifndef QFN_COMPARE_PREFILL_ACTIVATIONS_NO_MAIN

int main(int argc, char** argv) {
    using namespace qfn_act;
    std::vector<std::string> args(argv + 1, argv + argc);
    std::optional<Options> options;
    try {
        options = parse_options(args);
    } catch (const UsageError& error) {
        std::cerr << usage_text;
        std::cerr << "qfn_compare_prefill_activations: error: " << error.what() << '\n';
        return 2;
    }
    if (!options) {
        std::cout << usage_text;
        return 0;
    }
    if (!(options->abs_rms > 0) || !(options->rel_rms > 0)) {
        std::cerr << usage_text;
        std::cerr << "qfn_compare_prefill_activations: error: materiality thresholds must be positive\n";
        return 2;
    }
    std::cout << describe(*options) << '\n';
    std::cout.flush();
    try {
        run(options->root, options->layer, options->abs_rms, options->rel_rms, std::cout);
    } catch (const std::exception& error) {
        std::cout.flush();
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#endif // QFN_COMPARE_PREFILL_ACTIVATIONS_NO_MAIN
