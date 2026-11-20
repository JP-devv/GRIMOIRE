#include "b70/qfn_mtp_experts.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
constexpr int H = 64, I = 32;
const std::array<const char*, 3> names{"gate_proj", "up_proj", "down_proj"};

enum class Fault { none, packed_dtype, packed_shape, scale_dtype, scale_shape, shape_dtype,
                   shape_values, not_indexed, not_in_shard, zero_point, wrong_group };

void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void append_le(std::vector<uint8_t>& bytes, uint64_t v, int n) {
    for (int i = 0; i < n; ++i) bytes.push_back(uint8_t(v >> (i * 8)));
}

int code(int expert, int proj, int row, int col) {
    return (expert * 7 + proj * 5 + row * 3 + col + col / 32) % 16;
}

uint16_t scale_bits(int expert, int proj, int row, int group) {
    return uint16_t(0x3c01 + 32 * expert + 16 * proj + 2 * row + group);
}

struct Fixture {
    std::ostringstream index, headers[2];
    std::vector<uint8_t> data[2];

    void add(std::string name, int shard, const char* type, const std::string& shape,
             const std::vector<uint8_t>& bytes, bool indexed = true, bool present = true) {
        if (indexed) index << '"' << name << "\":\"part" << shard << ".safetensors\",";
        if (!present) return;
        auto& d = data[shard];
        headers[shard] << '"' << name << "\":{\"dtype\":\"" << type << "\",\"shape\":[" << shape
                       << "],\"data_offsets\":[" << d.size() << ',' << d.size() + bytes.size() << "]},";
        d.insert(d.end(), bytes.begin(), bytes.end());
    }

    void write(const fs::path& dir) const {
        auto members = index.str();
        members.pop_back();
        std::ofstream(dir / "model.safetensors.index.json") << "{\"weight_map\":{" << members << "}}";
        for (int s = 0; s < 2; ++s) {
            auto fields = headers[s].str();
            fields.back() = '}';
            std::string header = "{" + fields;
            while (header.size() % 8) header += ' ';
            auto f = std::ofstream(dir / ("part" + std::to_string(s) + ".safetensors"), std::ios::binary);
            for (int i = 0; i < 8; ++i) f.put(char(uint64_t(header.size()) >> (8 * i)));
            f << header;
            for (uint8_t v : data[s]) f.put(char(v));
            check(bool(f), "write synthetic safetensors");
        }
    }
};

void make_fixture(const fs::path& dir, Fault fault) {
    std::ofstream(dir / "config.json") << R"({"model_type":"qwen4_exp","text_config":{
        "model_type":"qwen4_exp_text","hidden_size":64,"moe_intermediate_size":32,"num_experts":2,
        "mtp":{"num_hidden_layers":1}},"quantization_config":{
        "quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{
        "mtp_routed_experts":{"format":"pack-quantized","targets":["RoutedExperts"],"input_activations":null,
        "weights":{"type":"int","num_bits":4,"strategy":"group","symmetric":true,"dynamic":false,
        "group_size":)" << (fault == Fault::wrong_group ? 64 : 32) << "}}}}}";
    Fixture f;
    // This index member belongs to another role. The routed loader must not open it.
    f.index << "\"mtp.fc_embedding.weight\":\"dense.safetensors\",";
    for (int e = 0; e < 2; ++e) for (int p = 0; p < 3; ++p) {
        const int n = p == 2 ? H : I, k = p == 2 ? I : H;
        const auto projection = "mtp.layers.0.mlp.experts." + std::to_string(e) + "." + names[p];
        const auto inject = e == 1 && p == 2 ? fault : Fault::none;
        std::vector<uint8_t> payload, scales, dims;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; c += 2)
                payload.push_back(uint8_t(code(e, p, r, c) | (code(e, p, r, c + 1) << 4)));
            for (int g = 0; g < k / 32; ++g) append_le(scales, scale_bits(e, p, r, g), 2);
        }
        append_le(dims, inject == Fault::shape_values ? k : n, 8);
        append_le(dims, inject == Fault::shape_values ? n : k, 8);
        auto matrix_shape = [](int rows, int cols, bool transpose) {
            if (transpose) std::swap(rows, cols);
            return std::to_string(rows) + ',' + std::to_string(cols);
        };
        f.add(projection + ".weight_packed", 0, inject == Fault::packed_dtype ? "F32" : "I32",
              matrix_shape(n, k / 8, inject == Fault::packed_shape), payload);
        f.add(projection + ".weight_scale", 1, inject == Fault::scale_dtype ? "F16" : "BF16",
              matrix_shape(n, k / 32, inject == Fault::scale_shape), scales,
              inject != Fault::not_indexed, inject != Fault::not_in_shard);
        f.add(projection + ".weight_shape", 0, inject == Fault::shape_dtype ? "F64" : "I64", "2", dims);
        if (inject == Fault::zero_point) f.add(projection + ".weight_zero_point", 1, "U8", "1", {8});
    }
    f.write(dir);
}

void check_loader() {
    char temp[] = "/tmp/qfn-mtp-int4.XXXXXX";
    check(::mkdtemp(temp) != nullptr, "mkdtemp");
    struct Scratch {
        fs::path path;
        ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
    } tmp{temp};
    make_fixture(tmp.path, Fault::none);
    b70::QfnMtpExperts draft;
    std::string err;
    check(draft.open(tmp.path.string(), err), "open fixture: " + err);
    check(draft.expert_count() == 2 && draft.hidden_size() == H && draft.intermediate_size() == I,
          "wrong draft geometry");
    for (int e = 0; e < 2; ++e) {
        b70::QfnMtpExpert expert;
        check(draft.read_expert(e, expert, err), "read test expert: " + err);
        const std::array<const b70::PackedWeight*, 3> weights{&expert.gate, &expert.up, &expert.down};
        for (int p = 0; p < 3; ++p) {
            const auto& w = *weights[p];
            const auto view = w.view();
            const int n = p == 2 ? H : I, k = p == 2 ? I : H;
            check(w.fmt == b70::Fmt::INT4 && w.N == n && w.K == k && w.row_bytes == k / 2 &&
                  w.row_scales == k / 32 && w.payload.size() == size_t(n * k / 2) &&
                  w.scales_raw.size() == size_t(n * k / 32 * 2) && w.zeros.size() == size_t(n * k / 32),
                  std::string(names[p]) + ": wrong matrix layout");
            for (int r = 0; r < n; ++r) {
                for (int g = 0; g < k / 32; ++g) {
                    const size_t i = size_t(r * (k / 32) + g);
                    const uint16_t bits = scale_bits(e, p, r, g);
                    check(w.zeros[i] == 8 && w.scales_raw[2*i] == uint8_t(bits) &&
                          w.scales_raw[2*i+1] == uint8_t(bits >> 8), "changed scales or zero metadata");
                }
                for (int c = 0; c < k; ++c) {
                    if (c % 2 == 0)
                        check(w.payload[size_t(r * k / 2 + c / 2)] ==
                              (code(e, p, r, c) | (code(e, p, r, c + 1) << 4)), "changed packed byte");
                    const float expected = float(code(e, p, r, c) - 8) *
                        b70::bf16_to_f32({scale_bits(e, p, r, c / 32)});
                    check(view.at(r, c) == expected, "dequantization differs at a loaded row");
                }
            }
        }
    }
    b70::QfnMtpExpert unused;
    check(!draft.read_expert(2, unused, err) && err.find("out of range") != std::string::npos,
          "accepted nonexistent expert");
    const std::array<std::pair<Fault, const char*>, 10> refusals{{
        {Fault::packed_dtype, ".weight_packed: incompatible"}, {Fault::packed_shape, ".weight_packed: incompatible"},
        {Fault::scale_dtype, ".weight_scale: incompatible"}, {Fault::scale_shape, ".weight_scale: incompatible"},
        {Fault::shape_dtype, ".weight_shape: incompatible"}, {Fault::shape_values, ".weight_shape: values"},
        {Fault::not_indexed, "missing indexed MTP tensor"}, {Fault::not_in_shard, "indexed MTP tensor not found"},
        {Fault::zero_point, "unsupported indexed MTP routed tensor"}, {Fault::wrong_group, "group-32 routed weights"},
    }};
    for (const auto& [fault, reason] : refusals) {
        make_fixture(tmp.path, fault);
        b70::QfnMtpExperts bad;
        err.clear();
        check(!bad.open(tmp.path.string(), err) && err.find(reason) != std::string::npos && bad.expert_count() == 0,
              std::string("expected refusal containing '") + reason + "', got '" + err + "'");
    }
}

int check_codes() {
    constexpr int group_size = 32;
    constexpr uint8_t offset_binary_zero = 8;
    constexpr uint8_t signed_s4 = 0xff;

    // Two little-endian I32 words: 0x76543210, 0xfedcba98. Every code
    // occurs exactly once; reversing nibbles or words must fail.
    constexpr std::array<uint8_t, 8> all_codes{
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
    };
    auto signed_codes = all_codes;
    for (auto& byte : signed_codes) byte ^= 0x88; // Flip the high bit of each nibble.
    for (int i = 0; i < 16; ++i) {
        const float expected = float(i - 8);
        if (b70::decode_int4(all_codes.data(), i, 1.0f, offset_binary_zero) != expected ||
            b70::decode_int4(all_codes.data(), i, 1.0f, signed_s4) == expected ||
            b70::decode_int4(signed_codes.data(), i, 1.0f, signed_s4) != expected) {
            std::fprintf(stderr, "all-code fixture failed at column %d (expected %g)\n", i, expected);
            return 1;
        }
    }

    // First two groups of expert 0 gate_proj, output row 0 in the MTP shard.
    constexpr std::array<uint8_t, 32> gate_payload{
        0x9d, 0x93, 0x66, 0x8b, 0x3b, 0x9d, 0xc8, 0x65,
        0x8a, 0x37, 0x74, 0x7d, 0xd8, 0x8a, 0xb7, 0x7f,
        0x79, 0x78, 0x76, 0x67, 0x54, 0x95, 0x81, 0x81,
        0x78, 0x74, 0xa9, 0x58, 0x57, 0x59, 0x98, 0x95,
    };
    constexpr std::array<int, 64> unpacked{
         5,  1, -5,  1, -2, -2,  3,  0,
         3, -5,  5,  1,  0,  4, -3, -2,
         2,  0, -1, -5, -4, -1,  5, -1,
         0,  5,  2,  0, -1,  3,  7, -1,
         1, -1,  0, -1, -2, -1, -1, -2,
        -4, -3, -3,  1, -7,  0, -7,  0,
         0, -1, -4, -1,  1,  2,  0, -3,
        -1, -3,  1, -3,  0,  1, -3,  1,
    };
    constexpr std::array<b70::bf16_t, 2> scales{{{0x3ba2}, {0x3c0b}}};
    constexpr std::array<uint8_t, 2> zeros{offset_binary_zero, offset_binary_zero};
    const b70::QuantWeight gate{b70::Fmt::INT4, 1, int(unpacked.size()), gate_payload.data(),
                               scales.data(), zeros.data(), int(gate_payload.size()), int(scales.size())};
    if ((1 << gate.int4_gshift()) != group_size) {
        std::fputs("draft gate fixture did not select group-32\n", stderr);
        return 1;
    }
    for (std::size_t k = 0; k < unpacked.size(); ++k) {
        const float scale = b70::bf16_to_f32(scales[k / group_size]);
        const float expected = float(unpacked[k]) * scale;
        const float actual = gate.at(0, int(k));
        if (actual != expected) {
            std::fprintf(stderr, "draft gate fixture failed at column %zu: expected %g, got %g\n",
                         k, expected, actual);
            return 1;
        }
    }
    std::puts("QFN draft INT4: all codes and two stored gate groups match with zero=8");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (check_codes() != 0) return 1;
        check_loader();
        std::puts("QFN external draft: gate/up/down bytes, BF16 scales, dequantization and refusals passed");
        if (argc == 2) {
            b70::QfnMtpExperts draft;
            std::string err;
            if (!draft.open(argv[1], err)) throw std::runtime_error(err);
            b70::QfnMtpExpert expert;
            for (int e : {0, draft.expert_count() - 1})
                if (!draft.read_expert(e, expert, err)) throw std::runtime_error(err);
            std::printf("QFN external draft: opened %d experts (H=%d I=%d), read both endpoint experts\n",
                        draft.expert_count(), draft.hidden_size(), draft.intermediate_size());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
