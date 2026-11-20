// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <vector>

namespace b70 {

// Frozen Flash-Next diagnostic only. Read rank 0's layer-zero mixer from
// the existing QFA1 stream, without reading unrelated activation payloads.
constexpr unsigned qfn_layer0_projection_width = 2560;
inline std::vector<float> read_qfn_layer0_projection(std::istream& in, unsigned start, unsigned rows) {
    if (!((start < 2560 && start % 512 == 0 && rows == 512) || (start == 2560 && rows == 146)))
        throw std::runtime_error("unexpected QFN layer-0 projection chunk");
    constexpr unsigned width = qfn_layer0_projection_width;
    constexpr uint32_t magic = 0x31414651, mixer_tap = 4;
    std::vector<float> out(size_t(rows) * width);
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    in.seekg(0);
    unsigned cursor = start;
    while (in && in.tellg() < end) {
        std::array<uint32_t, 7> h;
        if (!in.read(reinterpret_cast<char*>(h.data()), sizeof(h)))
            throw std::runtime_error("truncated QFN activation header");
        const auto [tag, tap, layer, rank, pos, count, w] = h;
        if (tag != magic || tap > 6 || layer >= 48 || rank != 0 ||
            pos < start || pos >= start + rows || !count || count > start + rows - pos || !w || w > 10240)
            throw std::runtime_error("invalid QFN activation header");
        const auto bytes = std::streamoff(count) * w * std::streamoff(sizeof(float));
        if (bytes > end - in.tellg()) throw std::runtime_error("truncated QFN activation payload");
        if (layer == 0 && tap == mixer_tap) {
            if (pos != cursor || w != width)
                throw std::runtime_error("invalid QFN layer-0 projection span");
            if (!in.read(reinterpret_cast<char*>(out.data() + size_t(pos - start) * width), bytes))
                throw std::runtime_error("cannot read QFN layer-0 projection span");
            cursor += count;
        } else {
            in.seekg(bytes, std::ios::cur);
        }
    }
    if (!in || cursor != start + rows ||
        !std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("incomplete or nonfinite QFN layer-0 projection capture");
    return out;
}

} // namespace b70
