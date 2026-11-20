"""Score paired MTP-off runs against predeclared raw-ID requests."""
import argparse
import json
from pathlib import Path
import re

import c1


def read_native(arm, prompt, output):
    prompt_line = f"prompt ids ({len(prompt)}): " + " ".join(map(str, prompt))
    emitted = None
    ranks = []
    phase_pattern = (
        rf"generation phases: prefill_tokens={len(prompt)} prefill_ms=(\d+\.\d+) "
        rf"decode_tokens={output - 1} decode_ms=(\d+\.\d+)"
    )
    for rank in range(4):
        lines = (arm / f"r{rank}.log").read_text().splitlines()
        if lines.count(prompt_line) != 1:
            raise ValueError(f"Rank {rank} prompt mismatch")
        greedy = [line.split(": ", 1)[1] for line in lines
                  if line.startswith(f"greedy ids ({output}): ")]
        if len(greedy) != 1:
            raise ValueError(f"Rank {rank} missing output receipt")
        tokens = c1.ids(greedy[0].split(), output)
        if emitted is not None and emitted != tokens:
            raise ValueError(f"Rank {rank} output mismatch")
        emitted = tokens
        phases = [re.fullmatch(phase_pattern, line) for line in lines
                  if line.startswith("generation phases:")]
        if len(phases) != 1 or phases[0] is None:
            raise ValueError(f"Rank {rank} phase receipt mismatch")
        ranks.append([float(value) / 1000 for value in phases[0].groups()])
    return emitted, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("native_arm", type=Path)
    parser.add_argument("reference_arm", type=Path)
    parser.add_argument("--expected-prefix", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    prompt, output = request["prompt"], request["max_tokens"]
    emitted, ranks = read_native(args.native_arm, prompt, output)
    if args.expected_prefix:
        expected = list(map(int, args.expected_prefix.read_text().split()))
        if emitted[:len(expected)] != expected:
            raise ValueError("Frozen output prefix mismatch")
    c1.PROMPT, c1.OUTPUT = len(prompt), output
    c1.fixture = lambda: (prompt, emitted)
    ref = args.reference_arm
    native = c1.result(*(max(values) for values in zip(*ranks)))
    vllm = c1.vllm(ref / "response.json", ref / "before.prom", ref / "after.prom")
    print(json.dumps({"native": native, "native_rank_seconds": ranks, "vllm": vllm}, indent=2))


if __name__ == "__main__":
    main()
