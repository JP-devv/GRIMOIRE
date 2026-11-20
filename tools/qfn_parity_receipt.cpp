// Compare the greedy receipts from the frozen, matched-chunk Flash-Next gate.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Ids = std::vector<int32_t>;
using Logit = std::pair<float, int32_t>;
const std::vector<int> chunk_rows{512, 512, 512, 512, 512, 146};

std::string suffix(const std::string& line, const std::string& prefix) {
    return line.compare(0, prefix.size(), prefix) == 0 ? line.substr(prefix.size()) : std::string{};
}

Ids read_ids(std::istream& file, int length) {
    std::string line, extra;
    if (!std::getline(file, line) || std::getline(file, extra)) throw std::runtime_error("expected one receipt line");
    const auto tokens = suffix(line, "greedy ids (" + std::to_string(length) + "):");
    std::istringstream in(tokens);
    Ids result;
    std::string raw;
    while (in >> raw) {
        size_t end = 0;
        const auto id = std::stoi(raw, &end);
        if (end != raw.size() || id < 0) throw std::runtime_error("invalid token ID");
        result.push_back(id);
    }
    if (!in.eof() || result.size() != size_t(length)) throw std::runtime_error("malformed greedy token receipt");
    return result;
}

void read_log(std::istream& in, bool head, std::vector<Logit>& logits) {
    std::string line;
    std::vector<int> chunks;
    int receipts = 0;
    while (std::getline(in, line)) {
        const size_t begin = line.find_first_not_of(' ');
        const std::string marker = "qwen4_exp prefill M=";
        if (begin != std::string::npos && line.compare(begin, marker.size(), marker) == 0 &&
            line.find(" at the end of the call", begin) != std::string::npos) {
            size_t end = 0;
            const auto tail = line.substr(begin + marker.size());
            const int value = std::stoi(tail, &end);
            if (end >= tail.size() || tail[end] != ',') throw std::runtime_error("malformed prefill chunk receipt");
            chunks.push_back(value);
        }
        if (!head) continue;
        const auto tail = suffix(line, "QFN logits pos=2706 ");
        if (tail.empty()) continue;
        ++receipts;
        std::istringstream payload(tail);
        std::string base, valid, token;
        if (!(payload >> base >> valid) || base.find("base=") != 0 || valid.find("valid=") != 0)
            throw std::runtime_error("malformed prefill logit header");
        while (payload >> token) {
            const size_t colon = token.find(':');
            if (colon == std::string::npos) throw std::runtime_error("missing logit separator");
            size_t end = 0;
            const int id = std::stoi(token, &end);
            if (end != colon || id < 0) throw std::runtime_error("invalid logit ID");
            const std::string raw = token.substr(colon + 1);
            const float value = std::stof(raw, &end);
            if (end != raw.size() || !std::isfinite(value)) throw std::runtime_error("invalid logit score");
            logits.emplace_back(value, id);
        }
    }
    if (chunks != chunk_rows || (head && receipts != 1))
        throw std::runtime_error("missing or unexpected six-chunk or position-2706 receipt");
}

Ids layout(const fs::path& root, const std::string& name, int length) {
    Ids result;
    std::vector<Logit> logits;
    for (int r = 0; r < 4; ++r) {
        const fs::path stem = root / (name + "-gate") / ("matched-r" + std::to_string(r));
        try {
            std::ifstream ids(stem.string() + ".ids"), log(stem.string() + ".log");
            const auto actual = read_ids(ids, length);
            if (r == 0) result = actual;
            else if (actual != result) throw std::runtime_error("rank-local greedy mismatch");
            // PP4 has a head only on its last rank; each TP4 arm must
            // supply logits from every vocabulary shard.
            read_log(log, name != "pp4" || r == 3, logits);
        } catch (const std::exception& e) {
            throw std::runtime_error(stem.string() + ": " + e.what());
        }
    }
    std::sort(logits.begin(), logits.end(), [](const Logit& a, const Logit& b) {
        return a.first == b.first ? a.second < b.second : a.first > b.first;
    });
    if (logits.size() < 2 || logits.front().second != result.front())
        throw std::runtime_error(name + ": first greedy ID does not match top logit");
    std::cout << name << ": all ranks completed five 512-row chunks and one 146-row chunk\n";
    std::cout << name << " pos=2706 first_token=" << result.front() << " logit="
              << std::setprecision(9) << logits.front().first << " margin="
              << (double(logits.front().first) - logits[1].first) << '\n';
    return result;
}

void self_test() {
    std::istringstream accepted("greedy ids (3): 2 3 5\n");
    if (read_ids(accepted, 3) != Ids{2, 3, 5}) throw std::runtime_error("token parsing regression");
    std::string body;
    for (const int n : chunk_rows) body += "    qwen4_exp prefill M=" + std::to_string(n) +
                                          ", sub-chunks of 512: free at the end of the call\n";
    std::istringstream log(body + "QFN logits pos=2706 base=0 valid=2/2 2:5.25 3:4\n");
    std::vector<Logit> logits;
    read_log(log, true, logits);
    if (logits != std::vector<Logit>{{5.25f, 2}, {4.f, 3}}) throw std::runtime_error("logit parsing regression");
    std::istringstream missing(body.substr(body.find('\n') + 1));
    try { read_log(missing, false, logits); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("missing chunk accepted");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            std::cout << "QFN parity receipt parser PASS\n";
            return 0;
        }
        const std::string mode = argc == 4 ? argv[1] : "";
        if (argc != 3 && mode != "--tp4" && mode != "--grouped")
            throw std::runtime_error("usage: qfn_parity_receipt [--tp4|--grouped] RUN_DIR 1|32");
        const std::string len = argv[argc == 4 ? 3 : 2];
        if (len != "1" && len != "32") throw std::runtime_error("expected output length 1 or 32");
        const auto root = argv[argc == 4 ? 2 : 1];
        const std::string left_name = mode == "--tp4" ? "host" : mode == "--grouped" ? "dense" : "pp4";
        const std::string right_name = mode == "--tp4" ? "peer" : mode == "--grouped" ? "grouped" : "tp4";
        const auto left = layout(root, left_name, std::stoi(len));
        const auto right = layout(root, right_name, std::stoi(len));
        const auto mismatch = std::mismatch(left.begin(), left.end(), right.begin());
        if (mismatch.first != left.end()) {
            std::cout << "FAIL: first differing generated token " << (mismatch.first - left.begin() + 1)
                      << " (" << (mode.empty() ? "PP4" : left_name) << ' ' << *mismatch.first
                      << ", " << (mode.empty() ? "TP4" : right_name) << ' ' << *mismatch.second << ")\n";
            return 1;
        }
        std::cout << "PASS: all " << len << " greedy IDs match\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "QFN parity receipt: " << e.what() << '\n';
        return 2;
    }
}
