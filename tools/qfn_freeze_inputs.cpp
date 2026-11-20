// SPDX-License-Identifier: GPL-3.0-or-later
//
// qfn_freeze_inputs: record indexed model views without reading or hashing
// tensor payloads.
//
//   qfn_freeze_inputs ROOT MODEL [MODEL...]
//
// ROOT is the allowed model root; MODEL paths are relative to it and may not
// resolve outside it. The document is a schema-1 receipt: sidecar digests, one
// entry per indexed shard with its header digest, the index's tensor and dtype
// census, and the indexer-tensor counts. Payload bytes are never read.
//
// A refusal prints "input receipt failed: ..." to stderr and exits 1, so a
// manifest step cannot silently record a partially validated checkpoint.
#include "b70/qfn_inputs.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(std::ostream& out) {
    out << "usage: qfn_freeze_inputs ROOT MODEL [MODEL...]\n"
           "  ROOT    allowed model root (shard aliases may not escape it)\n"
           "  MODEL   model directories relative to ROOT\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() == 1 && (args[0] == "-h" || args[0] == "--help")) {
        usage(std::cout);
        return 0;
    }
    if (args.size() < 2) {
        usage(std::cerr);
        return 2;
    }
    const std::filesystem::path root = args[0];
    try {
        std::vector<b70::QfnInputReceipt> receipts;
        receipts.reserve(args.size() - 1);
        for (std::size_t i = 1; i < args.size(); ++i)
            receipts.push_back(b70::qfn_input_receipt(root, root / args[i]));
        std::cout << b70::qfn_receipts_json(receipts) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "input receipt failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
