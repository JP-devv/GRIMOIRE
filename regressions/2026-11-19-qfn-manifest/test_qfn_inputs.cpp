// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_inputs.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;
struct TempDir {
    fs::path path;
    TempDir() {
        auto name = (fs::temp_directory_path() / "qfn-inputs.XXXXXX").string();
        if (char* dir = ::mkdtemp(name.data())) path = dir;
        else throw std::runtime_error("mkdtemp failed");
    }
    ~TempDir() { std::error_code error; fs::remove_all(path, error); }
};

void put(const fs::path& file, const std::string& bytes) {
    std::ofstream out(file, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    out.close();
    if (!out) throw std::runtime_error("fixture write failed");
}

void shard(const fs::path& file, uint64_t end, char payload = '\0') {
    const std::string header = R"({"body.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
      R"("body.indexer.weight":{"dtype":"BF16","shape":[2],"data_offsets":[4,)" + std::to_string(end) + "]}}";
    const uint64_t size = header.size();
    std::string encoded(8, '\0');
    for (int i = 0; i < 8; ++i) encoded[i] = char((size >> (8 * i)) & 255);
    put(file, encoded + header + std::string(8, payload));
}

void require(bool match, const char* description) {
    if (!match) throw std::runtime_error(description);
}
} // namespace

int main() {
    try {
        const TempDir dir;
        const auto root = dir.path / "models", model = root / "snapshot";
        fs::create_directories(model);
        const auto index = model / "model.safetensors.index.json", file = root / "physical.safetensors";
        const auto alias = model / "alias.safetensors";
        put(model / "config.json", "{\"indexer_budget\": 0}");
        fs::create_symlink(file, alias);
        const auto inspect = [&] { return b70::qfn_input_receipt(root, model); };
        const auto refuse = [&](const char* why) {
            try { (void)inspect(); }
            catch (const b70::QfnInputError& e) {
                if (std::string(e.what()).find(why) != std::string::npos) return;
                throw;
            }
            throw std::runtime_error(std::string("accepted invalid receipt: ") + why);
        };
        const auto select = [&](const std::string& tensor) {
            put(index, "{\"weight_map\":{\"" + tensor + "\":\"alias.safetensors\"}}");
        };
        shard(file, 8);
        select("body.weight");
        const auto first = inspect();
        require(first.tensors == 1 && first.indexed_bytes == 4 && first.indexed_indexer_tensors == 0 &&
                first.unindexed_indexer_tensors == 1 && first.shards.size() == 1 &&
                first.shards[0].resolved == "physical.safetensors", "index must be the allowlist through an alias");
        const auto document = b70::qfn_receipts_json({first});
        shard(file, 8, 'X');
        require(b70::qfn_receipts_json({inspect()}) == document, "payload bytes must not enter receipts");
        select("body.indexer.weight");
        const auto indexed = inspect();
        require(indexed.indexed_indexer_tensors == 1 && indexed.unindexed_indexer_tensors == 0,
                "index must allow selecting the physically present indexer");
        select("missing");
        refuse("not in shard");
        shard(file, 12);
        select("body.indexer.weight");
        refuse("invalid tensor span");
        put(index, R"({"weight_map":{"body.weight":"alias.safetensors","body.weight":"absent.safetensors"}})");
        refuse("invalid JSON");
        select("body.weight");
        const auto outside = dir.path / "outside.safetensors";
        fs::rename(file, outside);
        fs::remove(alias);
        fs::create_symlink(outside, alias);
        refuse("outside model root");
        using b70::qfn_inputs_detail::sha256_hex;
        require(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA-256 short-message vector");
        require(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
                    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                "SHA-256 padding-overflow vector");
        std::cout << "QFN input receipts: index allowlist, malformed spans, duplicate keys, alias containment and digests PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
