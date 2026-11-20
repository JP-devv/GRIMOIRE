// SPDX-License-Identifier: GPL-3.0-or-later
//
// Indexed model-view receipts: what the safetensors index says is loaded,
// proven with sidecar and header digests. Tensor payloads are never read or
// hashed, so a receipt for a 14 GB checkpoint costs a few hundred KB of I/O.
//
// The safetensors index is the allowlist. A shard may physically hold more
// tensors than the index selects (a shared or aliased shard); those extras are
// counted, never receipts. Paths resolve through symlinks and must stay under
// the model root, duplicate JSON keys and duplicate physical aliases are
// refused, and every indexed tensor must have a well-formed, non-overlapping
// span inside the shard's data segment.
#pragma once

#include "b70/json.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace b70 {

struct QfnInputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct QfnSidecar {
    std::string resolved;   // path relative to the model root
    std::string sha256;     // whole-file digest
};

struct QfnShard {
    std::string name;        // index shard name, a bare file name
    std::string resolved;    // aliased path relative to the model root
    std::string header_sha256;
    std::uint64_t file_bytes = 0;
    std::uint64_t indexed_bytes = 0;
    std::size_t tensors = 0;
    std::size_t unindexed_tensors = 0;
};

struct QfnInputReceipt {
    std::string model;                            // path relative to the model root
    std::map<std::string, QfnSidecar> sidecars;   // name -> digest
    std::vector<QfnShard> shards;                 // sorted by index shard name
    std::size_t tensors = 0;
    std::uint64_t indexed_bytes = 0;
    std::map<std::string, std::size_t> indexed_dtypes;
    std::size_t indexed_indexer_tensors = 0;
    std::size_t unindexed_indexer_tensors = 0;
};

namespace qfn_inputs_detail {

constexpr std::uint64_t max_header_bytes = 100000000;  // safetensors header ceiling

inline std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// FIPS 180-4 SHA-256. The project deliberately has no crypto dependency and
// the only digests here are over file headers and small sidecars.
class Sha256 {
public:
    void update(const void* data, std::size_t size) {
        const auto* p = static_cast<const unsigned char*>(data);
        total += size;
        while (size) {
            const std::size_t take = std::min(size, sizeof block - fill);
            std::copy(p, p + take, block + fill);
            fill += take;
            p += take;
            size -= take;
            if (fill == sizeof block) {
                compress(block);
                fill = 0;
            }
        }
    }
    std::string hex() {
        const std::uint64_t bits = total * 8;
        const unsigned char pad = 0x80;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (fill != 56) update(&zero, 1);
        unsigned char length[8];
        for (int i = 0; i < 8; ++i) length[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(length, sizeof length);
        std::string out;
        char buf[9];
        for (std::uint32_t word : h) {
            std::snprintf(buf, sizeof buf, "%08x", word);
            out += buf;
        }
        return out;
    }

private:
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    };

    void compress(const unsigned char* p) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
                   (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + s0 + maj;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64]{};
    std::size_t fill = 0;
    std::uint64_t total = 0;
};

inline std::string sha256_hex(const std::string& bytes) {
    Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.hex();
}

// Python json.dumps(..., ensure_ascii=True) escaping: non-ASCII escapes as
// \uXXXX so a receipt is byte-identical for the same checkpoint on any host.
inline void append_escaped(std::string& out, const std::string& s) {
    const char* hex = "0123456789abcdef";
    auto escape = [&](std::uint32_t cp) {
        out += "\\u";
        if (cp > 0xffff) {
            const std::uint32_t v = cp - 0x10000;
            cp = 0xd800 | (v >> 10);
            for (int shift : {12, 8, 4, 0}) out += hex[(cp >> shift) & 15];
            cp = 0xdc00 | (v & 0x3ff);
            for (int shift : {12, 8, 4, 0}) out += hex[(cp >> shift) & 15];
            return;
        }
        for (int shift : {12, 8, 4, 0}) out += hex[(cp >> shift) & 15];
    };
    out += '"';
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 32) {
                        out += "\\u00";
                        out += hex[c >> 4];
                        out += hex[c & 15];
                    } else {
                        out += char(c);
                    }
            }
            continue;
        }
        // Names come from validated JSON or from the filesystem; a byte that is
        // not a complete UTF-8 sequence is reported, not silently rewritten.
        std::uint32_t cp = 0;
        std::size_t width = 0;
        if ((c & 0xe0) == 0xc0) { cp = c & 0x1f; width = 2; }
        else if ((c & 0xf0) == 0xe0) { cp = c & 0x0f; width = 3; }
        else if ((c & 0xf8) == 0xf0) { cp = c & 0x07; width = 4; }
        if (!width || i + width > s.size())
            throw QfnInputError("path is not valid UTF-8: " + s);
        for (std::size_t j = 1; j < width; ++j) {
            const auto b = static_cast<unsigned char>(s[i + j]);
            if ((b & 0xc0) != 0x80) throw QfnInputError("path is not valid UTF-8: " + s);
            cp = (cp << 6) | (b & 0x3f);
        }
        escape(cp);
        i += width;
    }
    out += '"';
}

// Ordered JSON value with Python's sort_keys/indent=2 rendering. std::map
// supplies the output order, so the same receipt always prints identically.
struct Jv {
    enum class Kind { boolean, integer, string, array, object } kind = Kind::string;
    bool boolean = false;
    long long integer = 0;
    std::string string;
    std::vector<Jv> array;
    std::map<std::string, Jv> object;

    static Jv of(bool v) { Jv j; j.kind = Kind::boolean; j.boolean = v; return j; }
    static Jv of(long long v) { Jv j; j.kind = Kind::integer; j.integer = v; return j; }
    static Jv of(std::string v) { Jv j; j.kind = Kind::string; j.string = std::move(v); return j; }
    static Jv list() { Jv j; j.kind = Kind::array; return j; }
    static Jv map() { Jv j; j.kind = Kind::object; return j; }

    Jv& set(const std::string& key, Jv value) {
        object[key] = std::move(value);
        return *this;
    }
    Jv& push(Jv value) {
        array.push_back(std::move(value));
        return *this;
    }

    void write(std::string& out, int depth = 0) const {
        switch (kind) {
            case Kind::boolean: out += boolean ? "true" : "false"; break;
            case Kind::integer: out += std::to_string(integer); break;
            case Kind::string: append_escaped(out, string); break;
            case Kind::array: {
                if (array.empty()) { out += "[]"; break; }
                out += "[\n";
                for (std::size_t i = 0; i < array.size(); ++i) {
                    if (i) out += ",\n";
                    out.append(std::size_t(2 * (depth + 1)), ' ');
                    array[i].write(out, depth + 1);
                }
                out += '\n';
                out.append(std::size_t(2 * depth), ' ');
                out += ']';
                break;
            }
            case Kind::object: {
                if (object.empty()) { out += "{}"; break; }
                out += "{\n";
                bool first = true;
                for (const auto& entry : object) {
                    if (!first) out += ",\n";
                    first = false;
                    out.append(std::size_t(2 * (depth + 1)), ' ');
                    append_escaped(out, entry.first);
                    out += ": ";
                    entry.second.write(out, depth + 1);
                }
                out += '\n';
                out.append(std::size_t(2 * depth), ' ');
                out += '}';
                break;
            }
        }
    }
};

inline std::filesystem::path canonical_checked(const std::filesystem::path& path) {
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(path, ec);
    if (ec) throw QfnInputError("cannot resolve " + path.string() + ": " + ec.message());
    return resolved;
}

// Component-wise prefix test, so /models2 is not treated as inside /models.
inline bool within(const std::filesystem::path& root, const std::filesystem::path& path) {
    auto r = root.begin();
    auto p = path.begin();
    for (; r != root.end(); ++r, ++p)
        if (p == path.end() || *r != *p) return false;
    return true;
}

inline std::string relative_to(const std::filesystem::path& root,
                               const std::filesystem::path& path) {
    auto p = path.begin();
    for (auto r = root.begin(); r != root.end(); ++r, ++p) {}
    std::filesystem::path rel;
    for (; p != path.end(); ++p) rel /= *p;
    if (rel.empty()) rel = ".";
    return rel.generic_string();
}

inline std::filesystem::path resolve_input(const std::filesystem::path& root,
                                           const std::filesystem::path& path) {
    const auto resolved = canonical_checked(path);
    if (!within(root, resolved))
        throw QfnInputError("outside model root: " + path.string() + " -> " + resolved.string());
    std::error_code ec;
    if (!std::filesystem::is_regular_file(resolved, ec))
        throw QfnInputError("not a regular file: " + resolved.string());
    return resolved;
}

inline std::string read_all(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw QfnInputError("cannot open " + path.string());
    std::string out;
    char buffer[1 << 16];
    while (in.read(buffer, sizeof buffer) || in.gcount() > 0)
        out.append(buffer, std::size_t(in.gcount()));
    if (in.bad()) throw QfnInputError("read failed: " + path.string());
    return out;
}

// JSON integers only: type(value) is int in the reference, so "4.0" and true
// are not offsets. Values outside int64 cannot be inside any data segment.
inline bool json_int(const Json& v, long long& out) {
    if (v.kind != Json::Kind::Number || v.literal.empty()) return false;
    if (v.literal.find_first_of(".eE") != std::string::npos) return false;
    std::size_t used = 0;
    try {
        out = std::stoll(v.literal, &used);
    } catch (const std::exception&) {
        return false;
    }
    return used == v.literal.size();
}

inline bool shard_name_ok(const std::string& name) {
    constexpr std::string_view suffix = ".safetensors";
    if (name.empty() || name == "." || name == "..") return false;
    if (name.find('/') != std::string::npos || name.find('\0') != std::string::npos) return false;
    return name.size() > suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline Jv shard_json(const QfnShard& shard) {
    return Jv::map()
        .set("file_bytes", Jv::of(static_cast<long long>(shard.file_bytes)))
        .set("header_sha256", Jv::of(shard.header_sha256))
        .set("indexed_bytes", Jv::of(static_cast<long long>(shard.indexed_bytes)))
        .set("name", Jv::of(shard.name))
        .set("resolved", Jv::of(shard.resolved))
        .set("tensors", Jv::of(static_cast<long long>(shard.tensors)))
        .set("unindexed_tensors", Jv::of(static_cast<long long>(shard.unindexed_tensors)));
}

} // namespace qfn_inputs_detail

// Mandatory sidecars come first; the rest are receipts only when present.
inline const std::vector<std::string>& qfn_sidecar_names() {
    static const std::vector<std::string> names = {
        "config.json", "model.safetensors.index.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    };
    return names;
}

// Reads sidecars and shard headers only. Throws QfnInputError on any refusal.
inline QfnInputReceipt qfn_input_receipt(const std::filesystem::path& root_path,
                                         const std::filesystem::path& model_path) {
    using namespace qfn_inputs_detail;
    const auto root = canonical_checked(root_path);
    const auto model = canonical_checked(model_path);
    std::error_code ec;
    if (!within(root, model) || !std::filesystem::is_directory(model, ec))
        throw QfnInputError("not a model directory under " + root.string() + ": " +
                            model.string());

    QfnInputReceipt receipt;
    std::map<std::string, std::string> sidecar_bytes;
    const auto& names = qfn_sidecar_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        const auto path = model / name;
        if (i > 1) {
            std::error_code exists_ec, link_ec;
            if (!std::filesystem::exists(path, exists_ec) &&
                !std::filesystem::is_symlink(std::filesystem::symlink_status(path, link_ec)))
                continue;
        }
        const auto resolved = resolve_input(root, path);
        std::string raw = read_all(resolved);
        receipt.sidecars[name] = QfnSidecar{relative_to(root, resolved), sha256_hex(raw)};
        sidecar_bytes[name] = std::move(raw);
    }

    const auto index_bytes = sidecar_bytes.at("model.safetensors.index.json");
    Json index;
    try {
        index = Json::parse(index_bytes);
    } catch (const std::invalid_argument& error) {
        // Duplicate object keys fail here: the parser refuses to overwrite one.
        throw QfnInputError("invalid JSON in model.safetensors.index.json: " +
                            std::string(error.what()));
    }
    const Json* weights = index.kind == Json::Kind::Object ? index.find("weight_map") : nullptr;
    if (!weights || weights->kind != Json::Kind::Object || weights->object.empty())
        throw QfnInputError("missing or empty weight_map: " + model.string());

    std::map<std::string, std::set<std::string>> by_shard;
    for (const auto& entry : weights->object) {
        const Json& value = entry.second;
        if (value.kind != Json::Kind::String || !shard_name_ok(value.string))
            throw QfnInputError("invalid safetensors shard for " + entry.first + ": " +
                                value.dump());
        by_shard[value.string].insert(entry.first);
    }
    receipt.tensors = weights->object.size();
    for (const auto& entry : weights->object)
        if (entry.first.find(".indexer.") != std::string::npos)
            ++receipt.indexed_indexer_tensors;

    std::set<std::filesystem::path> seen;
    for (const auto& entry : by_shard) {
        const std::string& name = entry.first;
        const std::set<std::string>& selected = entry.second;
        const auto resolved = resolve_input(root, model / name);
        if (!seen.insert(resolved).second)
            throw QfnInputError("multiple indexed aliases for " + resolved.string());

        std::ifstream in(resolved, std::ios::binary);
        if (!in) throw QfnInputError("cannot open " + resolved.string());
        const std::uint64_t file_bytes = std::filesystem::file_size(resolved);
        unsigned char prefix[8];
        if (!in.read(reinterpret_cast<char*>(prefix), sizeof prefix))
            throw QfnInputError("short safetensors prefix: " + name);
        std::uint64_t header_bytes = 0;
        for (int i = 7; i >= 0; --i) header_bytes = (header_bytes << 8) | prefix[i];
        if (!(header_bytes > 0 && header_bytes <= max_header_bytes &&
              header_bytes <= file_bytes - sizeof prefix))
            throw QfnInputError("invalid safetensors header length: " + name);
        std::string raw(std::size_t(header_bytes), '\0');
        if (in.read(raw.data(), std::streamsize(raw.size())).gcount() !=
            std::streamsize(raw.size()))
            throw QfnInputError("short safetensors header: " + name);

        const std::uint64_t data_bytes = file_bytes - sizeof prefix - header_bytes;
        Json header;
        try {
            header = Json::parse(raw);
        } catch (const std::invalid_argument& error) {
            throw QfnInputError("invalid JSON header in " + name + ": " +
                                std::string(error.what()));
        }
        if (header.kind != Json::Kind::Object)
            throw QfnInputError("invalid safetensors header: " + name);
        std::set<std::string> unindexed;
        for (const auto& tensor : header.object)
            if (tensor.first != "__metadata__" && !selected.count(tensor.first))
                unindexed.insert(tensor.first);

        std::vector<std::pair<long long, long long>> spans;
        QfnShard shard;
        shard.name = name;
        shard.resolved = relative_to(root, resolved);
        shard.file_bytes = file_bytes;
        shard.header_sha256 = sha256_hex(raw);
        shard.tensors = selected.size();
        shard.unindexed_tensors = unindexed.size();
        for (const auto& key : selected) {
            const Json* desc = header.find(key);
            if (!desc) throw QfnInputError("tensor " + key + " not in shard " + name);
            long long first = 0, last = 0;
            const Json* offsets = desc->kind == Json::Kind::Object ? desc->find("data_offsets")
                                                                  : nullptr;
            const bool span_ok = offsets && offsets->kind == Json::Kind::Array &&
                offsets->array.size() == 2 && json_int(offsets->array[0], first) &&
                json_int(offsets->array[1], last) && first >= 0 && first <= last &&
                static_cast<std::uint64_t>(last) <= data_bytes;
            if (!span_ok)
                throw QfnInputError("invalid tensor span: " + name + ":" + key);
            const Json* dtype = desc->find("dtype");
            if (!dtype || dtype->kind != Json::Kind::String || dtype->string.empty())
                throw QfnInputError("invalid tensor dtype: " + name + ":" + key);
            ++receipt.indexed_dtypes[dtype->string];
            if (first != last) spans.emplace_back(first, last);
        }
        std::sort(spans.begin(), spans.end());
        long long end = 0;
        for (const auto& span : spans) {
            if (span.first < end)
                throw QfnInputError("overlapping indexed tensor spans: " + name);
            end = span.second;
            shard.indexed_bytes += std::uint64_t(span.second - span.first);
        }
        for (const auto& key : unindexed)
            if (key.find(".indexer.") != std::string::npos) ++receipt.unindexed_indexer_tensors;
        receipt.indexed_bytes += shard.indexed_bytes;
        receipt.shards.push_back(std::move(shard));
    }
    receipt.model = relative_to(root, model);
    return receipt;
}

inline qfn_inputs_detail::Jv qfn_receipt_value(const QfnInputReceipt& receipt) {
    using qfn_inputs_detail::Jv;
    Jv shards = Jv::list();
    for (const auto& shard : receipt.shards) shards.push(qfn_inputs_detail::shard_json(shard));
    Jv sidecars = Jv::map();
    for (const auto& entry : receipt.sidecars)
        sidecars.set(entry.first, Jv::map().set("resolved", Jv::of(entry.second.resolved))
                                       .set("sha256", Jv::of(entry.second.sha256)));
    Jv dtypes = Jv::map();
    for (const auto& entry : receipt.indexed_dtypes)
        dtypes.set(entry.first, Jv::of(static_cast<long long>(entry.second)));
    return Jv::map()
        .set("indexed_bytes", Jv::of(static_cast<long long>(receipt.indexed_bytes)))
        .set("indexed_dtypes", std::move(dtypes))
        .set("indexed_indexer_tensors",
             Jv::of(static_cast<long long>(receipt.indexed_indexer_tensors)))
        .set("model", Jv::of(receipt.model))
        .set("shards", std::move(shards))
        .set("sidecars", std::move(sidecars))
        .set("tensors", Jv::of(static_cast<long long>(receipt.tensors)))
        .set("unindexed_indexer_tensors",
             Jv::of(static_cast<long long>(receipt.unindexed_indexer_tensors)));
}

// One model receipt as text, for a single-model caller or a test.
inline std::string qfn_receipt_json(const QfnInputReceipt& receipt) {
    std::string out;
    qfn_receipt_value(receipt).write(out);
    return out;
}

// The complete receipt document: schema 1, no trailing newline.
inline std::string qfn_receipts_json(const std::vector<QfnInputReceipt>& receipts) {
    using qfn_inputs_detail::Jv;
    Jv models = Jv::list();
    for (const auto& receipt : receipts) models.push(qfn_receipt_value(receipt));
    std::string out;
    Jv::map()
        .set("models", std::move(models))
        .set("schema", Jv::of(1LL))
        .set("tensor_payloads_hashed", Jv::of(false))
        .write(out);
    return out;
}

} // namespace b70
