"""Score paired MTP-off runs against predeclared raw-ID requests."""
import argparse
import json
import math
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


def read_reference(directory, model, prompt, emitted):
    response = json.loads((directory / "response.json").read_text())
    choice, = response["choices"]
    usage = response["usage"]
    if (choice.get("prompt_token_ids", response.get("prompt_token_ids")) != prompt
            or choice["token_ids"] != emitted or choice["finish_reason"] != "length"
            or usage["prompt_tokens"] != len(prompt) or usage["completion_tokens"] != len(emitted)
            or usage["prompt_tokens_details"]["cached_tokens"] != 0):
        raise ValueError("Reference IDs, lengths or cache receipt disagree")

    def sample(path, metric):
        rows = re.findall(r"^" + re.escape(metric) + r"\{([^}\n]*)\}\s+([^\s]+)\s*$",
                          path.read_text(), re.M)
        matches = [float(value) for labels, value in rows
                   if f'model_name="{model}"' in labels.split(",")]
        if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] < 0:
            raise ValueError(f"Expected one valid {metric} series")
        return matches[0]

    before, after = directory / "before.prom", directory / "after.prom"
    durations = []
    for phase in ("prefill", "decode"):
        metric = f"vllm:request_{phase}_time_seconds"
        if sample(after, metric + "_count") - sample(before, metric + "_count") != 1:
            raise ValueError(f"Expected one {phase} observation")
        durations.append(sample(after, metric + "_sum") - sample(before, metric + "_sum"))
    return durations


def score(prefill_s, decode_s, prompt_tokens, output_tokens):
    if not all(math.isfinite(seconds) and seconds > 0 for seconds in (prefill_s, decode_s)):
        raise ValueError("Invalid phase duration")
    return {"prefill_s": prefill_s, "decode_s": decode_s,
            "cold_prefill_tok_s": prompt_tokens / prefill_s,
            "decode_tok_s": (output_tokens - 1) / decode_s}


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
    native = score(*(max(values) for values in zip(*ranks)), len(prompt), output)
    vllm = score(*read_reference(args.reference_arm, request["model"], prompt, emitted),
                 len(prompt), output)
    print(json.dumps({"native": native, "native_rank_seconds": ranks, "vllm": vllm}, indent=2))


if __name__ == "__main__":
    main()
