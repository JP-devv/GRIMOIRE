// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_layer0_projection.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

void record(std::ostream& out, unsigned tap, unsigned layer, unsigned rank,
            unsigned pos, unsigned rows, unsigned width, float value) {
    const std::array<uint32_t, 7> header{0x31414651, tap, layer, rank, pos, rows, width};
    out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    const std::vector<float> payload(size_t(rows) * width, value);
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(float));
}

template<class F> void refuses(F fill, const char* message) {
    std::stringstream capture(std::ios::in | std::ios::out | std::ios::binary);
    fill(capture);
    try { (void)b70::read_qfn_layer0_projection(capture, 2560, 146); }
    catch (const std::runtime_error&) { return; }
    require(false, message);
}
} // namespace

int main() {
    for (const auto [start, rows] : std::array<std::array<unsigned, 2>, 6>{{
            {0,512}, {512,512}, {1024,512}, {1536,512}, {2048,512}, {2560,146}}}) {
        std::stringstream capture(std::ios::in | std::ios::out | std::ios::binary);
        record(capture, 5, 0, 0, start, rows, 2560, -9); // MoE decoy, same width
        record(capture, 4, 1, 0, start, rows, 2560, -8); // layer-1 mixer decoy
        record(capture, 4, 0, 0, start, rows-1, 2560, float(start + 1));
        record(capture, 4, 0, 0, start+rows-1, 1, 2560, float(start + 2));
        const auto output = b70::read_qfn_layer0_projection(capture, start, rows);
        require(output.size() == rows * 2560 && output.front() == float(start + 1) &&
                output[(rows-1)*2560 - 1] == float(start + 1) &&
                output[(rows-1)*2560] == float(start + 2) && output.back() == float(start + 2),
                "only ordered layer-0 projection rows may be replayed");
    }
    refuses([](auto& f) { record(f, 4, 0, 0, 2560, 145, 2560, 1); }, "missing last row accepted");
    refuses([](auto& f) { record(f, 4, 0, 0, 2561, 145, 2560, 1); }, "wrong row offset accepted");
    refuses([](auto& f) { record(f, 4, 0, 1, 2560, 146, 2560, 1); }, "wrong PP rank accepted");
    refuses([](auto& f) { record(f, 4, 0, 0, 2560, 146, 1280, 1); }, "wrong width accepted");
    refuses([](auto& f) { record(f, 4, 0, 0, 2560, 146, 2560,
                               std::numeric_limits<float>::infinity()); }, "nonfinite input accepted");
    refuses([](auto& f) { record(f, 4, 0, 0, 2560, 146, 2560, 1); f.put('x'); }, "partial header accepted");
    refuses([](auto& f) { record(f, 4, 0, 0, 2560, 146, 2560, 1);
                        record(f, 4, 0, 0, 2560, 146, 2560, 2); }, "duplicate mixer span accepted");
    std::cout << "QFN layer-0 projection replay: six chunks, tap selection, and refusals PASS\n";
}
