// SPDX-License-Identifier: GPL-3.0-or-later
//
// Host-only self-test for tools/qfn_compare_prefill_activations.cpp.
//
// Build and run (no GPU, no Python):
//   g++ -std=c++17 -O2 -Wall -Wextra -Iinclude regressions/2026-11-19-qfn-prefill-activations/test_qfn_prefill_activations.cpp
//       -o bin/test_qfn_prefill_activations
//   ./bin/test_qfn_prefill_activations
//
// Covers:
//   * ownership        - only the PP4 stage owning a layer is compared against,
//                        even when every other stage holds decoy records;
//                        stage 0..3 wiring across all 48 layers in full mode.
//   * split spans      - PP4 writes whole-chunk records while TP4 ranks split
//                        them differently, so rows must be joined by absolute
//                        position, not by record boundary.
//   * malformed payload- truncated header, truncated payload, bad magic, wrong
//                        rank/layer/tap, overlapping rows, width drift, missing
//                        tail, width mismatch, missing key, nonfinite values,
//                        missing file, plus CLI/threshold handling.

#define QFN_COMPARE_PREFILL_ACTIVATIONS_NO_MAIN
#include "../../tools/qfn_compare_prefill_activations.cpp"

#include <cstdlib>
#include <sstream>
#include <unistd.h>

namespace {

using namespace qfn_act;

int failures = 0;

void require(bool ok, const std::string& what) {
    if (ok) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

void require_contains(const std::string& haystack, const std::string& needle,
                      const std::string& what) {
    require(haystack.find(needle) != std::string::npos,
            what + " (missing \"" + needle + "\" in output:\n" + haystack + ")");
}

std::string temp_root(const std::string& tag) {
    static int counter = 0;
    const std::filesystem::path base = std::filesystem::temp_directory_path() /
                                       ("qfn-act-test-" + std::to_string(::getpid()) + "-" +
                                        std::to_string(counter++) + "-" + tag);
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base / "pp4");
    std::filesystem::create_directories(base / "tp4");
    return base.string();
}

struct FixtureRecord {
    std::uint32_t tap = 0;
    std::uint32_t layer = 12;
    std::uint32_t rank = 0;
    std::uint32_t pos = 0;
    std::uint32_t count = 1;
    std::uint32_t width = 10240;
    std::uint32_t magic_field = magic;
};

void write_record(std::ofstream& out, const FixtureRecord& r, std::vector<float>& scratch) {
    unsigned char header[header_size];
    store_u32le(header + 0, r.magic_field);
    store_u32le(header + 4, r.tap);
    store_u32le(header + 8, r.layer);
    store_u32le(header + 12, r.rank);
    store_u32le(header + 16, r.pos);
    store_u32le(header + 20, r.count);
    store_u32le(header + 24, r.width);
    out.write(reinterpret_cast<const char*>(header), std::streamsize(header_size));
    scratch.resize(std::size_t(r.count) * r.width);
    std::vector<unsigned char> payload(scratch.size() * 4);
    for (std::size_t i = 0; i < scratch.size(); ++i) store_f32le(payload.data() + i * 4, scratch[i]);
    out.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size()));
}

// One rank file of one chunk. `value` fills the payload; `split` chooses how the
// rows are chopped into records.
struct RankFile {
    std::uint32_t chunk_start = 0;
    std::uint32_t chunk_rows = 1;
    std::uint32_t rank = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> taps = {{1, 10240}, {4, 2560}, {5, 2560}, {6, 10240}};
    std::uint32_t layer = 12;
    std::uint32_t split = 1; // rows per record; 0 = whole chunk in one record
    double value = 1.0;
    std::uint32_t value_tap = 99;   // tap whose value differs, 99 = none
    std::uint32_t value_start = 99; // chunk start whose value differs, 99 = any
    double special_value = 1.0;
    bool nan_at_first_row = false;
    bool empty = false;
    bool append = false; // keep records already written to this rank file
};

// Writes every row of every record of one rank file; `value` is the exact FP32
// magnitude a row is filled with.

void write_rank_file(const std::string& dir, const std::string& tag, const RankFile& spec) {
    const std::string path = record_path(std::filesystem::path(dir), tag.c_str(), spec.rank,
                                         spec.chunk_start);
    std::ofstream out(path, std::ios::binary | (spec.append ? std::ios::app : std::ios::trunc));
    require(bool(out), "open fixture " + path);
    if (spec.empty) return;
    std::vector<float> scratch;
    for (const auto& tap : spec.taps) {
        const std::uint32_t step = spec.split == 0 ? spec.chunk_rows : spec.split;
        for (std::uint32_t off = 0; off < spec.chunk_rows; off += step) {
            const std::uint32_t count = std::min(step, spec.chunk_rows - off);
            FixtureRecord record;
            record.tap = tap.first;
            record.width = tap.second;
            record.layer = spec.layer;
            record.rank = spec.rank;
            record.pos = spec.chunk_start + off;
            record.count = count;
            const bool special = tap.first == spec.value_tap &&
                                 (spec.value_start == 99 || spec.value_start == spec.chunk_start);
            const double value = special ? spec.special_value : spec.value;
            scratch.assign(std::size_t(count) * tap.second, float(value));
            if (spec.nan_at_first_row && tap.first == 1 && off == 0) scratch.front() = std::nanf("");
            write_record(out, record, scratch);
        }
    }
}

// Default fixture: chunk (0,2) then (2,1), layer 12 (owner = 1), four taps.
// pp4 writes only rank 1 with whole-chunk records; tp4 writes all ranks, even
// ranks in single-row records and odd ranks in whole-chunk records.
void build_split_fixture(const std::string& root, bool mutate_tp4_rank2 = true) {
    const std::vector<Chunk> chunks = {{0, 2}, {2, 1}};
    for (const Chunk& chunk : chunks) {
        for (std::uint32_t rank = 0; rank < pp_world; ++rank) {
            RankFile spec;
            spec.chunk_start = chunk.start;
            spec.chunk_rows = chunk.rows;
            spec.rank = rank;
            spec.split = 0;
            spec.empty = rank != 1; // only the owner writes
            write_rank_file(root, "pp4", spec);
        }
        for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
            RankFile spec;
            spec.chunk_start = chunk.start;
            spec.chunk_rows = chunk.rows;
            spec.rank = rank;
            spec.split = rank % 2 == 0 ? 1 : 0;
            spec.value_tap = mutate_tp4_rank2 && rank == 2 ? 4 : 99;
            spec.value_start = mutate_tp4_rank2 && rank == 2 ? 2 : 99;
            spec.special_value = 1.02; // numpy float32 ones + 0.02
            write_rank_file(root, "tp4", spec);
        }
    }
}

std::string run_text(const std::string& root, std::optional<std::uint32_t> layer, double abs_floor,
                     double rel_floor, const std::vector<Chunk>& chunks, Result* out_result,
                     bool* threw, std::string* message) {
    std::ostringstream text;
    if (threw) *threw = false;
    try {
        Result result = run(std::filesystem::path(root), layer, abs_floor, rel_floor, text, chunks);
        if (out_result) *out_result = result;
    } catch (const std::exception& error) {
        if (!threw) throw;
        *threw = true;
        if (message) *message = error.what();
    }
    return text.str();
}

void expect_error(const std::string& root, const std::string& needle, const std::string& what,
                  std::optional<std::uint32_t> layer = 12,
                  const std::vector<Chunk>& chunks = {{0, 2}}) {
    bool threw = false;
    std::string message;
    run_text(root, layer, 1e-4, 0.01, chunks, nullptr, &threw, &message);
    require(threw, what + ": expected an error");
    if (threw) require_contains(message, needle, what);
}

// ------------------------------------------------------------- the checks --

void test_split_spans_and_ownership_of_rows() {
    const std::string root = temp_root("split");
    build_split_fixture(root);
    const std::vector<Chunk> chunks = {{0, 2}, {2, 1}};
    Result result;
    const std::string text = run_text(root, 12, 1e-4, 0.01, chunks, &result, nullptr, nullptr);

    require(result.compared_rows == 4u * 4u * 3u,
            "split spans: 4 taps x 4 TP ranks x 3 rows compared, got " +
                std::to_string(result.compared_rows));
    require(result.summary.size() == 16, "split spans: 16 (layer, tap, rank) summaries");
    for (const auto& entry : result.summary)
        require(entry.second.rows == 3, "split spans: each summary covers all 3 rows");

    require(result.first_nonzero.has_value(), "split spans: one nonzero row expected");
    require(result.first_material.has_value(), "split spans: one material row expected");
    if (result.first_nonzero && result.first_material) {
        const Hit& nz = *result.first_nonzero;
        const Hit& mat = *result.first_material;
        require(nz.layer == 12 && nz.tap == 4 && nz.pos == 2 && nz.rank == 2,
                "split spans: first_nonzero is layer 12 tap mixer pos 2 tp_rank 2");
        require(mat.layer == 12 && mat.tap == 4 && mat.pos == 2 && mat.rank == 2,
                "split spans: first_material is layer 12 tap mixer pos 2 tp_rank 2");
        const double expected = double(1.02f) - 1.0;
        require(std::abs(mat.rms - expected) <= 1e-12 * expected,
                "split spans: material rms is (1.02f - 1.0f), got " + format_g8(mat.rms));
        require(std::abs(mat.rel_rms - expected) <= 1e-12,
                "split spans: material rel_rms is 0.02, got " + format_g8(mat.rel_rms));
    }

    const Stats& moe = result.summary.at(SummaryKey{12, 5, 0});
    require(moe.rows == 3 && moe.max_rms == 0.0 && moe.material_rows == 0,
            "split spans: moe rank 0 is exact across both chunks");
    const Stats& mixer = result.summary.at(SummaryKey{12, 4, 2});
    require(mixer.material_rows == 1 && mixer.rows == 3,
            "split spans: mixer rank 2 has exactly one material row");

    require_contains(text, "checked [0,2) vs all TP4 ranks", "split spans: chunk 0 progress");
    require_contains(text, "checked [2,3) vs all TP4 ranks", "split spans: chunk 1 progress");
    require_contains(text, "layer=12 tap=moe tp_rank=0 rows=3 max_rms=0 max_rel_rms=0 material_rows=0",
                     "split spans: exact summary line");
    require_contains(text, "first_material layer=12 tap=mixer pos=2 tp_rank=2 rms=0.019999981 rel_rms=0.019999981",
                     "split spans: first_material line");
    require_contains(text, "first_nonzero layer=12 tap=mixer pos=2 tp_rank=2 rms=0.019999981 rel_rms=0.019999981",
                     "split spans: first_nonzero line");
    // Summary order is (layer, tap, rank), taps ascending: entry, mixer, moe, exit.
    require(text.find("tap=entry tp_rank=0") < text.find("tap=mixer tp_rank=0") &&
                text.find("tap=mixer tp_rank=0") < text.find("tap=moe tp_rank=0") &&
                text.find("tap=moe tp_rank=0") < text.find("tap=exit tp_rank=0"),
            "split spans: summary sorted by layer, tap, rank");
    std::filesystem::remove_all(root);
}

void test_owner_selection_ignores_decoys() {
    // Same shape, but every non-owner PP4 stage holds a decoy record for the
    // same key with a large value. Only PP4 stage 1 may be used as reference.
    const std::string root = temp_root("owner");
    const std::vector<Chunk> chunks = {{0, 2}, {2, 1}};
    build_split_fixture(root, false);
    for (const Chunk& chunk : chunks) {
        for (std::uint32_t rank = 0; rank < pp_world; ++rank) {
            if (rank == 1) continue;
            RankFile spec;
            spec.chunk_start = chunk.start;
            spec.chunk_rows = chunk.rows;
            spec.rank = rank;
            spec.split = 0;
            spec.value = 7.0; // decoy
            write_rank_file(root, "pp4", spec);
        }
    }
    Result result;
    const std::string text = run_text(root, 12, 1e-4, 0.01, chunks, &result, nullptr, nullptr);
    require(!result.first_material.has_value(), "owner: decoy PP4 stages do not create diffs");
    for (const auto& entry : result.summary)
        require(entry.second.max_rms == 0.0, "owner: every (tap, rank) is exact against stage 1");
    require_contains(text, "first_material none", "owner: nothing material");
    require_contains(text, "first_nonzero none", "owner: nothing nonzero");
    std::filesystem::remove_all(root);
}

void test_full_mode_stage_wiring() {
    // Full scan: no --layer, so layers 0..47 must resolve to stage layer/12 and
    // every stage holds decoys for layers it does not own.
    const std::string root = temp_root("full");
    const std::vector<Chunk> chunks = {{0, 1}};
    for (std::uint32_t rank = 0; rank < pp_world; ++rank) {
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
            RankFile spec;
            spec.chunk_start = 0;
            spec.chunk_rows = 1;
            spec.rank = rank;
            spec.layer = layer;
            spec.split = 0;
            const bool owned = layer / layers_per_stage == rank;
            // Owned rows carry layer+1, decoys carry 42.
            spec.value = owned ? double(layer + 1) : 42.0;
            spec.taps = {{1, 10240}, {6, 10240}};
            if (layer == 0) spec.taps.push_back({0, 10240});
            spec.append = layer != 0;
            write_rank_file(root, "pp4", spec);
        }
    }
    for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
            RankFile spec;
            spec.chunk_start = 0;
            spec.chunk_rows = 1;
            spec.rank = rank;
            spec.layer = layer;
            spec.split = 0;
            spec.value = double(layer + 1);
            spec.taps = {{1, 10240}, {6, 10240}};
            if (layer == 0) spec.taps.push_back({0, 10240});
            spec.append = layer != 0;
            write_rank_file(root, "tp4", spec);
        }
    }
    Result result;
    const std::string text = run_text(root, std::nullopt, 1e-4, 0.01, chunks, &result, nullptr,
                                      nullptr);
    const std::size_t expected_rows = (std::size_t(2) * layer_count + 1) * tp_world;
    require(result.compared_rows == expected_rows,
            "full: 97 keys x 4 ranks compared, got " + std::to_string(result.compared_rows));
    require(result.summary.size() == 97u * tp_world, "full: one summary per key and rank");
    for (const auto& entry : result.summary)
        require(entry.second.max_rms == 0.0,
                "full: layer " + std::to_string(std::get<0>(entry.first)) + " tap " +
                    tap_names[std::get<1>(entry.first)] + " rank " +
                    std::to_string(std::get<2>(entry.first)) + " matches its owning stage");
    require(!result.first_nonzero.has_value(), "full: no nonzero rows");
    require(select_keys(1).size() == 6, "full: layer 1 selects entry/mixer/moe/exit + PLE pair");
    require(select_keys(std::uint32_t(0)).size() == 5, "full: layer 0 adds the embed tap");
    require(select_keys(std::nullopt).size() == 97, "full: 48 layers x 2 taps + layer 0 embed");
    std::filesystem::remove_all(root);
}

void test_malformed_payload() {
    const std::vector<Chunk> chunks = {{0, 2}};

    { // truncated header: a stray partial header at the tail
        const std::string root = temp_root("trunc-hdr");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "tp4", 0, 0);
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.write("QFA1", 4);
        out.close();
        expect_error(root, "tp4/r0-p0.bin: truncated header", "malformed: truncated header",
                     std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
    { // truncated payload: the last record's body is four bytes short
        const std::string root = temp_root("trunc-payload");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "tp4", 2, 2);
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 4);
        expect_error(root, "tp4/r2-p2.bin: invalid record", "malformed: truncated payload",
                     std::uint32_t(12), {{0, 2}, {2, 1}});
        std::filesystem::remove_all(root);
    }
    { // bad magic
        const std::string root = temp_root("magic");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "pp4", 1, 0);
        std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
        const unsigned char zero[4] = {0, 0, 0, 0};
        io.write(reinterpret_cast<const char*>(zero), 4);
        io.close();
        expect_error(root, "pp4/r1-p0.bin: invalid record (0, ", "malformed: bad magic",
                     std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
    { // rank field disagreeing with the file name
        const std::string root = temp_root("rank");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "tp4", 0, 0);
        std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
        unsigned char rank_field[4];
        store_u32le(rank_field, 3);
        io.seekp(12);
        io.write(reinterpret_cast<const char*>(rank_field), 4);
        io.close();
        expect_error(root, "invalid record", "malformed: rank header mismatch", std::uint32_t(12),
                     chunks);
        std::filesystem::remove_all(root);
    }
    { // layer out of range and tap out of range
        const std::string root = temp_root("layer-tap");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "tp4", 1, 0);
        std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
        unsigned char field[4];
        store_u32le(field, layer_count); // 48
        io.seekp(8);
        io.write(reinterpret_cast<const char*>(field), 4);
        io.close();
        expect_error(root, "invalid record", "malformed: layer out of range", std::uint32_t(12),
                     chunks);

        const std::string tap_root = temp_root("tap");
        build_split_fixture(tap_root);
        const std::string tap_path = record_path(std::filesystem::path(tap_root), "tp4", 1, 0);
        std::fstream tap_io(tap_path, std::ios::binary | std::ios::in | std::ios::out);
        store_u32le(field, 7);
        tap_io.seekp(4);
        tap_io.write(reinterpret_cast<const char*>(field), 4);
        tap_io.close();
        expect_error(tap_root, "invalid record", "malformed: tap out of range", std::uint32_t(12),
                     chunks);
        std::filesystem::remove_all(root);
        std::filesystem::remove_all(tap_root);
    }
    { // overlapping rows: the same position twice
        const std::string root = temp_root("overlap");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "tp4", 0, 0);
        std::vector<float> values;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        FixtureRecord record;
        record.tap = 1;
        record.layer = 12;
        record.rank = 0;
        record.pos = 0;
        record.count = 1;
        record.width = 10240;
        values.assign(10240, 1.0f);
        write_record(out, record, values); // pos 0
        write_record(out, record, values); // pos 0 again
        out.close();
        expect_error(root, "overlapping/gapped rows or width drift at (12, 1)",
                     "malformed: overlapping rows", std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
    { // width drift inside one key
        const std::string root = temp_root("drift");
        build_split_fixture(root);
        const std::string path = record_path(std::filesystem::path(root), "pp4", 1, 0);
        std::vector<float> values;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        FixtureRecord record;
        record.tap = 1;
        record.layer = 12;
        record.rank = 1;
        record.pos = 0;
        record.count = 1;
        record.width = 10240;
        values.assign(10240, 1.0f);
        write_record(out, record, values);
        record.pos = 1;
        record.width = 2560; // drift
        values.assign(2560, 1.0f);
        write_record(out, record, values);
        out.close();
        expect_error(root, "overlapping/gapped rows or width drift at (12, 1)",
                     "malformed: width drift", std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
    { // missing tail: a 2-row chunk covered by a single row
        const std::string root = temp_root("tail");
        build_split_fixture(root);
        for (std::uint32_t rank = 0; rank < tp_world; ++rank) {
            const std::string path = record_path(std::filesystem::path(root), "tp4", rank, 0);
            std::vector<float> values;
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            FixtureRecord record;
            record.tap = 1;
            record.layer = 12;
            record.rank = rank;
            record.pos = 0;
            record.count = 1;
            record.width = 10240;
            values.assign(10240, 1.0f);
            write_record(out, record, values);
            out.close();
        }
        expect_error(root, "missing tail at (12, 1)", "malformed: missing tail", std::uint32_t(12),
                     chunks);
        std::filesystem::remove_all(root);
    }
    { // width mismatch against the frozen tap width
        const std::string root = temp_root("width");
        build_split_fixture(root);
        // Rewrite every tap-4 record of the PP4 owner with width 10240.
        const std::string path = record_path(std::filesystem::path(root), "pp4", 1, 0);
        std::vector<float> values;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        for (const auto& tap : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
                 {1, 10240}, {4, 10240}, {5, 2560}, {6, 10240}}) {
            FixtureRecord record;
            record.tap = tap.first;
            record.layer = 12;
            record.rank = 1;
            record.pos = 0;
            record.count = 2;
            record.width = tap.second;
            values.assign(std::size_t(2) * tap.second, 1.0f);
            write_record(out, record, values);
        }
        out.close();
        expect_error(root, "width mismatch for (12, 4) at 0", "malformed: width mismatch",
                     std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
    { // missing key from the PP4 owner and from one TP4 rank
        const std::string root = temp_root("missing");
        build_split_fixture(root);
        std::filesystem::remove(record_path(std::filesystem::path(root), "pp4", 1, 0));
        std::filesystem::remove(record_path(std::filesystem::path(root), "pp4", 1, 2));
        // Rebuild the owner without the exit tap.
        for (const Chunk& chunk : chunks) {
            RankFile spec;
            spec.chunk_start = chunk.start;
            spec.chunk_rows = chunk.rows;
            spec.rank = 1;
            spec.split = 0;
            spec.taps = {{1, 10240}, {4, 2560}, {5, 2560}};
            write_rank_file(root, "pp4", spec);
        }
        expect_error(root, "missing (12, 6) at chunk 0: PP4 owner 1 or one of TP4",
                     "malformed: missing PP4 key", std::uint32_t(12), chunks);

        const std::string tp_root = temp_root("missing-tp");
        build_split_fixture(tp_root);
        for (const Chunk& chunk : chunks) {
            RankFile spec;
            spec.chunk_start = chunk.start;
            spec.chunk_rows = chunk.rows;
            spec.rank = 3;
            spec.split = 0;
            spec.taps = {{1, 10240}, {4, 2560}, {6, 10240}};
            write_rank_file(tp_root, "tp4", spec);
        }
        expect_error(tp_root, "missing (12, 5) at chunk 0: PP4 owner 1 or one of TP4",
                     "malformed: missing TP4 key", std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
        std::filesystem::remove_all(tp_root);
    }
    { // nonfinite activation
        const std::string root = temp_root("nan");
        const std::vector<Chunk> nan_chunks = {{0, 2}, {2, 1}};
        build_split_fixture(root);
        RankFile spec;
        spec.chunk_start = 0;
        spec.chunk_rows = 2;
        spec.rank = 1;
        spec.split = 0;
        spec.nan_at_first_row = true;
        write_rank_file(root, "tp4", spec);
        expect_error(root, "nonfinite activation", "malformed: nonfinite activation",
                     std::uint32_t(12), nan_chunks);
        std::filesystem::remove_all(root);
    }
    { // absent rank file
        const std::string root = temp_root("absent");
        build_split_fixture(root);
        std::filesystem::remove(record_path(std::filesystem::path(root), "pp4", 1, 0));
        expect_error(root, "pp4/r1-p0.bin: cannot open", "malformed: missing rank file",
                     std::uint32_t(12), chunks);
        std::filesystem::remove_all(root);
    }
}

void test_thresholds_and_cli() {
    const std::string root = temp_root("threshold");
    build_split_fixture(root);
    const std::vector<Chunk> chunks = {{0, 2}, {2, 1}};
    // Raising the absolute floor above the 0.02 delta hides the difference.
    Result result;
    run_text(root, 12, 0.5, 0.01, chunks, &result, nullptr, nullptr);
    require(!result.first_material.has_value(), "threshold: abs floor 0.5 hides the material row");
    require(result.compared_rows == 48, "threshold: rows are still compared");
    // Raising the relative floor above 0.02 hides it as well.
    Result relative;
    run_text(root, 12, 1e-4, 0.5, chunks, &relative, nullptr, nullptr);
    require(!relative.first_material.has_value(), "threshold: rel floor 0.5 hides the material row");

    const auto options = parse_options({"regressions/root", "--layer", "12", "--abs-rms=0.0",
                                        "--rel-rms", "2"});
    require(options.has_value(), "cli: option set parses");
    if (options) {
        require(options->root == "regressions/root", "cli: positional root");
        require(options->layer == std::uint32_t(12), "cli: --layer");
        require(options->abs_rms == 0.0 && options->rel_rms == 2.0, "cli: thresholds");
    }
    bool rejected = false;
    try {
        parse_options({"--layer", "48", "root"});
    } catch (const UsageError&) {
        rejected = true;
    }
    require(rejected, "cli: --layer 48 is rejected");
    rejected = false;
    try {
        parse_options({"--bogus"});
    } catch (const UsageError&) {
        rejected = true;
    }
    require(rejected, "cli: unknown flag is rejected");
    rejected = false;
    try {
        parse_options({"--layer", "1"});
    } catch (const UsageError&) {
        rejected = true;
    }
    require(rejected, "cli: missing root is rejected");
    require(!parse_options({"--help"}).has_value(), "cli: --help returns no options");

    const Options described;
    require(describe(described) ==
                "material iff RMS delta >= 0.0001 and relative RMS delta >= 0.01; ordered by layer, "
                "tap, absolute token, TP rank",
            "cli: threshold banner matches the Python scanner");
    std::filesystem::remove_all(root);
}

void test_frozen_geometry() {
    const std::vector<Chunk>& chunks = frozen_chunks();
    require(chunks.size() == 6, "geometry: six frozen chunks");
    std::uint32_t cursor = 0;
    for (const Chunk& chunk : chunks) {
        require(chunk.start == cursor, "geometry: chunks are contiguous");
        cursor += chunk.rows;
    }
    require(cursor == 2706, "geometry: 2706 frozen prefill tokens");
    require(chunks.back().rows == 146, "geometry: the last chunk is the 146-token tail");
    require(layer_count / layers_per_stage == pp_world, "geometry: 48 layers over 4 PP stages");
}

} // namespace

int main() {
    const auto ple_keys = select_keys(1);
    const auto next_keys = select_keys(2);
    for (unsigned tap : {2u, 3u}) {
        require(std::count(ple_keys.begin(), ple_keys.end(), Key{1, tap}) == 1,
                "frozen ple_layer_ids=[2] means zero-based layer 1");
        require(std::count(next_keys.begin(), next_keys.end(), Key{2, tap}) == 0,
                "zero-based layer 2 has no PLE taps");
    }
    const std::array<std::pair<const char*, void (*)()>, 6> tests = {{
        {"frozen geometry", test_frozen_geometry},
        {"split spans", test_split_spans_and_ownership_of_rows},
        {"owner selection", test_owner_selection_ignores_decoys},
        {"full mode stage wiring", test_full_mode_stage_wiring},
        {"malformed payload", test_malformed_payload},
        {"thresholds and cli", test_thresholds_and_cli},
    }};
    for (const auto& test : tests) {
        try {
            test.second();
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << test.first << " threw: " << error.what() << '\n';
            ++failures;
        }
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "qfn_compare_prefill_activations: all checks passed\n";
    return 0;
}
