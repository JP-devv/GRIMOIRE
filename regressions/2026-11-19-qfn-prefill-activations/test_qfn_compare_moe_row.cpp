// SPDX-License-Identifier: GPL-3.0-or-later
//
// Native (host-only, std C++17) equivalent of
// regressions/2026-11-19-qfn-prefill-activations/test_moe_row.py and of the
// common-input boundary checks in benchmarks/qfn/compare_moe_row.py:main().
//
// It includes the port itself rather than linking it, so the test drives the
// exact code paths the tool runs and cannot drift from copy-pasted logic.
//
// Build and run from the worktree root, with the same flags as the host
// Makefile (CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude):
//   mkdir -p bin
//   g++ -std=c++17 -O2 -Wall -Wextra -Iinclude regressions/2026-11-19-qfn-prefill-activations/test_qfn_compare_moe_row.cpp -o bin/test_qfn_compare_moe_row
//   ./bin/test_qfn_compare_moe_row

#define QFN_COMPARE_MOE_ROW_NO_MAIN
#include "../../tools/qfn_compare_moe_row.cpp"

#include <random>
#include <sstream>

namespace {

using namespace qfn_moe_row;

int checks = 0;

void require(bool ok, const std::string& what) {
    ++checks;
    if (ok) return;
    std::cerr << "FAIL " << what << '\n';
    std::exit(1);
}

void require_contains(const std::string& haystack, const std::string& needle, const std::string& what) {
    require(haystack.find(needle) != std::string::npos,
            what + "\n  missing: " + needle + "\n  output:\n" + haystack);
}

void require_absent(const std::string& haystack, const std::string& needle, const std::string& what) {
    require(haystack.find(needle) == std::string::npos,
            what + "\n  unexpected: " + needle + "\n  output:\n" + haystack);
}

template <typename Body>
void require_throws(Body&& body, const std::string& needle, const std::string& what) {
    ++checks;
    try {
        body();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) return;
        std::cerr << "FAIL " << what << "\n  wrong message: " << error.what() << "\n  wanted: " << needle << '\n';
        std::exit(1);
    }
    std::cerr << "FAIL " << what << "\n  no exception raised\n";
    std::exit(1);
}

// capture_stdout(): the port prints the report on std::cout, so a test can
// assert on the same strings the Python test greps out of redirect_stdout.
template <typename Body>
std::string capture_stdout(Body&& body) {
    std::ostringstream log;
    std::streambuf* const previous = std::cout.rdbuf(log.rdbuf());
    try {
        body();
    } catch (...) {
        std::cout.rdbuf(previous);
        throw;
    }
    std::cout.rdbuf(previous);
    return log.str();
}

struct TempDir {
    fs::path path;

    TempDir() {
        std::random_device device;
        std::mt19937_64 generator(device());
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::ostringstream name;
            name << "qfn-moe-row-" << std::hex << generator();
            const fs::path candidate = fs::temp_directory_path() / name.str();
            std::error_code ec;
            if (fs::create_directory(candidate, ec)) {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("cannot create a temporary directory");
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

// Little-endian record writer, the counterpart of the '<f4' / '<i4' dumps.
template <typename T>
void put(const fs::path& path, const std::vector<T>& values) {
    static_assert(std::is_same<T, float>::value || std::is_same<T, int>::value, "put() writes float or int records");
    std::vector<unsigned char> bytes(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits;
        if constexpr (std::is_same<T, float>::value) {
            const float value = values[i];
            std::memcpy(&bits, &value, sizeof bits);
        } else {
            const int value = values[i];
            std::memcpy(&bits, &value, sizeof bits);
        }
        for (int byte = 0; byte < 4; ++byte) bytes[4 * i + std::size_t(byte)] = (bits >> (8 * byte)) & 0xffu;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(path.string() + ": cannot open for writing");
    if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// The Python fixture: experts 1..385 including one per TP4 rank boundary,
// plus 511 just outside the cut, and reductions that are exact in float32 so a
// correct reconstruction prints rms=0.
void write_fixture(const fs::path& root) {
    const std::vector<int> ids = {1, 128, 257, 384, 2, 129, 260, 385, 3, 132};
    std::vector<float> logits(kExperts, 0.0f);
    for (std::size_t slot = 0; slot < kTopK; ++slot) logits[std::size_t(ids[slot])] = float(kTopK - slot);
    logits[511] = 0.75f;
    const std::vector<float> weights(kTopK, 0.125f);
    const std::vector<float> ones(kHidden, 1.0f);

    std::vector<float> contributions[10];
    for (std::size_t slot = 0; slot < kTopK; ++slot) {
        contributions[slot].assign(kHidden, 0.0f);
        for (std::size_t i = 0; i < kHidden; ++i) contributions[slot][i] = float(slot + 1) / 16.0f;
    }
    std::vector<float> output(kHidden, 0.0f);  // Python: sum(contributions) + row
    for (std::size_t slot = 0; slot < kTopK; ++slot)
        for (std::size_t i = 0; i < kHidden; ++i) output[i] = output[i] + contributions[slot][i];
    for (std::size_t i = 0; i < kHidden; ++i) output[i] = output[i] + ones[i];

    for (const std::string layout : {"pp4", "tp4"}) {
        const fs::path directory = root / layout;
        fs::create_directories(directory);
        const bool tensor_parallel = layout == "tp4";
        for (int rank = 0; rank < (tensor_parallel ? kWorld : 1); ++rank) {
            const auto file = [&](const std::string& name) { return record_path(directory, 1, 361, rank, name); };
            std::vector<int> local(kTopK);
            std::vector<float> routed(kHidden, 0.0f);
            for (std::size_t slot = 0; slot < kTopK; ++slot) {
                const int global = ids[slot];
                const bool resident = !tensor_parallel || global / kExpertsPerRank == rank;
                local[slot] = tensor_parallel ? (resident ? global - rank * kExpertsPerRank : -1) : global;
                if (!resident) continue;
                put(file("expert-" + std::to_string(global)), contributions[slot]);
                for (std::size_t i = 0; i < kHidden; ++i) routed[i] += contributions[slot][i];
            }
            put(file("global"), ids);
            put(file("local"), local);
            put(file("logits"), logits);
            put(file("weights"), weights);
            put(file("native-input"), ones);
            put(file("input"), ones);
            put(file("routed"), routed);
            std::vector<float> shared = ones;
            if (tensor_parallel)
                for (float& value : shared) value /= 4.0f;
            put(file("shared"), shared);
            std::vector<float> partial(kHidden);
            for (std::size_t i = 0; i < kHidden; ++i) partial[i] = routed[i] + shared[i];
            put(file("partial"), partial);
            put(file("reduced"), output);
        }
    }
}

// test_moe_row.py: cutoff margin, resident ownership, float32 reconstruction.
void test_global_expert_ownership_and_reduction() {
    TempDir temp;
    write_fixture(temp.path);

    Arm arm;
    std::string out = capture_stdout([&] { arm = compare_arm(temp.path, 1, 361); });
    require(!out.empty(), "compare_arm prints a report");

    require_contains(out, "pp4 rank=0 top_k=[1, 128, 257, 384, 2, 129, 260, 385, 3, 132]", "captured PP4 top-k");
    require_contains(out, "cutoff: 132:1 vs 511:0.75; margin=0.25", "10th/11th logit cutoff and margin");
    require_contains(out, "local_ids=[1, 128, 257, 384, 2, 129, 260, 385, 3, 132] remap=OK", "PP4 keeps global IDs");
    require_contains(out, "tp4 rank=2 top_k=[1, 128, 257, 384, 2, 129, 260, 385, 3, 132]", "TP4 captures global IDs");
    require_contains(out, "local_ids=[-1, -1, 1, -1, -1, -1, 4, -1, -1, -1] remap=OK", "TP4 resident remap");
    require_contains(out, "routed vs captured slots: rms=0 rel=0 max_abs=0", "routed equals its slot sum");
    require_contains(out, "partial vs routed+shared: rms=0 rel=0 max_abs=0", "partial equals routed+shared");
    require_contains(out, "global expert=257 owner=2 weighted PP4 vs TP4: rms=0", "expert 257 owner and value");
    require_contains(out, "TP4 sum vs rank=3 reduced: rms=0", "TP4 FP32 reduction reconstruction");
    require_contains(out, "PP4 vs sum of TP4 shared: rms=0", "shared deltas");
    require_contains(out, "only PP4: []; only TP4: []", "contribution sets agree");

    // Every TP4 rank must carry the whole, identical routing decision.
    require(top_k_equal(arm), "TP4 ranks agree with PP4 on top-k");

    // A wrong resident index still owns the expert: remap, not routing, breaks.
    const fs::path mapping = record_path(temp.path / "tp4", 1, 361, 2, "local");
    std::vector<int> local = read_i32(mapping, kTopK);
    require(local[2] == 1, "rank 2 addresses global expert 257 as local 1");
    local[2] += 1;
    put(mapping, local);
    out = capture_stdout([&] { compare_arm(temp.path, 1, 361); });
    require_contains(out, "remap=MISMATCH", "misaddressed resident expert is reported");

    // Truncated receipts must fail size validation, not be tolerated.
    fs::resize_file(record_path(temp.path / "tp4", 1, 361, 2, "expert-257"), 8);
    require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); }, "wrong byte size",
                   "truncated expert receipt is rejected");
}

// The reconstruction check must not be vacuous: one perturbed FP32 slot has to
// show up in the TP4 sum against the reduced row. The fold is shared by every
// rank comparison, so the perturbation appears against all four ranks while the
// untouched reduced rows keep their exact PP4 agreement.
void test_reduction_reconstruction_detects_a_perturbation() {
    TempDir temp;
    write_fixture(temp.path);
    std::string out = capture_stdout([&] { compare_arm(temp.path, 1, 361); });
    for (int rank = 0; rank < kWorld; ++rank)
        require_contains(out, "TP4 sum vs rank=" + std::to_string(rank) + " reduced: rms=0 rel=0 max_abs=0",
                         "intact capture reconstructs exactly on rank " + std::to_string(rank));

    const fs::path partial = record_path(temp.path / "tp4", 1, 361, 3, "partial");
    std::vector<float> values = read_f32(partial, kHidden);
    values[0] += 1.0e-3f;
    put(partial, values);

    out = capture_stdout([&] { compare_arm(temp.path, 1, 361); });
    for (int rank = 0; rank < kWorld; ++rank) {
        const std::string line = "TP4 sum vs rank=" + std::to_string(rank) + " reduced: ";
        require_contains(out, line + "rms=", "rank " + std::to_string(rank) + " sum is still reported");
        require_absent(out, line + "rms=0 rel=0 max_abs=0",
                       "perturbed partial breaks the rank " + std::to_string(rank) + " sum");
    }
    require_contains(out, "PP4 vs TP4 rank=3: input rms=0 rel=0 max_abs=0; logits rms=0 rel=0 max_abs=0; output rms=0 rel=0 max_abs=0",
                     "unperturbed reduced rows still match PP4 exactly");
}

// test_moe_row.py's boundary path plus main()'s common-input byte equality.
void test_common_input_boundary() {
    TempDir temp;
    write_fixture(temp.path / "native");
    write_fixture(temp.path / "common");

    std::string out = capture_stdout([&] { run(temp.path, 1, 361); });
    require_contains(out, "=== Native inputs ===", "native section header");
    require_contains(out, "=== Same captured PP4 input ===", "common section header");
    require_contains(out, "BOUNDARY native_topk_equal=True common_topk_equal=True", "boundary report");

    // A common-input PP4 row that differs from the native bytes must be named.
    {
        TempDir broken;
        write_fixture(broken.path / "native");
        write_fixture(broken.path / "common");
        put(record_path(broken.path / "common/pp4", 1, 361, 0, "input"), std::vector<float>(kHidden, 2.0f));
        require_throws([&] { capture_stdout([&] { run(broken.path, 1, 361); }); },
                       "common PP4 did not receive the exact PP4 input bytes", "common PP4 input equality");
    }

    // Same for a TP4 rank, reported with its rank label.
    {
        TempDir broken;
        write_fixture(broken.path / "native");
        write_fixture(broken.path / "common");
        put(record_path(broken.path / "common/tp4", 1, 361, 1, "input"), std::vector<float>(kHidden, 2.0f));
        require_throws([&] { capture_stdout([&] { run(broken.path, 1, 361); }); },
                       "common TP4-1 did not receive the exact PP4 input bytes", "common TP4 rank input equality");
    }

    // The common PP4 arm must reproduce its own native reference bit for bit,
    // even when the input bytes are identical.
    {
        TempDir broken;
        write_fixture(broken.path / "native");
        write_fixture(broken.path / "common");
        const fs::path logits_path = record_path(broken.path / "common/pp4", 1, 361, 0, "logits");
        std::vector<float> logits = read_f32(logits_path, kExperts);
        logits[511] = 0.751f;  // still outside the cut, so routing is unchanged
        put(logits_path, logits);
        require_throws([&] { capture_stdout([&] { run(broken.path, 1, 361); }); },
                       "common-input PP4 failed to reproduce its native reference", "common PP4 native reproduction");
    }
}

// Routing-order and data validation shared with the Python reader.
void test_routing_and_data_validation() {
    {
        // Exact tie at the cut: the GPU keeps the lower global expert ID, so a
        // capture that lists the higher one is rejected.
        TempDir temp;
        write_fixture(temp.path);
        const fs::path logits_path = record_path(temp.path / "pp4", 1, 361, 0, "logits");
        std::vector<float> logits = read_f32(logits_path, kExperts);
        logits[400] = 1.0f;  // ties with expert 132 at the 10th slot
        put(logits_path, logits);
        std::vector<int> ids = read_i32(record_path(temp.path / "pp4", 1, 361, 0, "global"), kTopK);
        ids[9] = 400;
        put(record_path(temp.path / "pp4", 1, 361, 0, "global"), ids);
        require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); },
                       "captured logit order disagrees with GPU top-k", "top-k tie resolution");
    }
    {
        // Duplicate global ID.
        TempDir temp;
        write_fixture(temp.path);
        std::vector<int> ids = read_i32(record_path(temp.path / "pp4", 1, 361, 0, "global"), kTopK);
        ids[1] = ids[0];
        put(record_path(temp.path / "pp4", 1, 361, 0, "global"), ids);
        require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); }, "invalid global IDs",
                       "duplicate global ID is rejected");
    }
    {
        // Out-of-range global ID.
        TempDir temp;
        write_fixture(temp.path);
        std::vector<int> ids = read_i32(record_path(temp.path / "pp4", 1, 361, 0, "global"), kTopK);
        ids[0] = int(kExperts);
        put(record_path(temp.path / "pp4", 1, 361, 0, "global"), ids);
        require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); }, "invalid global IDs",
                       "out-of-range global ID is rejected");
    }
    {
        // Non-finite activation.
        TempDir temp;
        write_fixture(temp.path);
        std::vector<float> logits = read_f32(record_path(temp.path / "pp4", 1, 361, 0, "logits"), kExperts);
        logits[5] = std::numeric_limits<float>::quiet_NaN();
        put(record_path(temp.path / "pp4", 1, 361, 0, "logits"), logits);
        require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); }, "non-finite data",
                       "non-finite data is rejected");
    }
    {
        // Wrong byte size on the routing receipt itself.
        TempDir temp;
        write_fixture(temp.path);
        put(record_path(temp.path / "pp4", 1, 361, 0, "weights"), std::vector<float>(kTopK - 1, 0.125f));
        require_throws([&] { capture_stdout([&] { compare_arm(temp.path, 1, 361); }); }, "wrong byte size",
                       "short weight receipt is rejected");
    }
}

// The report must be byte-identical for repeated runs of the same capture.
void test_report_is_deterministic() {
    TempDir temp;
    write_fixture(temp.path / "native");
    write_fixture(temp.path / "common");
    const std::string first = capture_stdout([&] { run(temp.path, 1, 361); });
    const std::string second = capture_stdout([&] { run(temp.path, 1, 361); });
    require(first == second, "repeated runs print the same report");
    require_contains(first, "rms=0 rel=0 max_abs=0", "self-comparison is exactly zero");
}

}  // namespace

int main() {
    const auto balanced = delta(Row{1, 1}, Row{0, 2});
    require(balanced.rms == 1 && balanced.rel == 1 && balanced.max_abs == 1,
            "signed deltas must not cancel in the RMS");
    test_global_expert_ownership_and_reduction();
    std::cout << "ok: global-ID ownership, cutoff margin, FP32 reduction reconstruction\n";
    test_reduction_reconstruction_detects_a_perturbation();
    std::cout << "ok: FP32 reduction reconstruction is not vacuous\n";
    test_common_input_boundary();
    std::cout << "ok: common-input exact byte equality and native reproduction\n";
    test_routing_and_data_validation();
    std::cout << "ok: top-k tie resolution and record validation\n";
    test_report_is_deterministic();
    std::cout << "ok: deterministic report\n";
    std::cout << "PASS " << checks << " checks" << '\n';
    return 0;
}
