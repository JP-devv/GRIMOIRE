"""Read worker provenance and device activity from vLLM XPU profiler traces."""
import argparse
import gzip
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    traces = {}
    captured_ranks = set()
    for path in sorted(args.trace_dir.rglob("*.pt.trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        device = [event for event in events
                  if event.get("cat") == "kernel" and event.get("dur", 0) > 0]
        if not device:
            raise SystemExit(f"Missing XPU kernel intervals in {path}")
        labels = [event["name"] for event in events
                  if event.get("name", "").startswith("qfn/r") and event.get("ph") == "X"]
        steps = [re.fullmatch(r"qfn/r(\d+)/worker-step/rows=(\d+)", label)
                 for label in labels]
        steps = [step for step in steps if step is not None]
        ranks = {int(step[1]) for step in steps}
        if len(ranks) != 1 or captured_ranks.intersection(ranks):
            raise SystemExit(f"Expected a unique QFN model worker in {path}")
        rank = ranks.pop()
        captured_ranks.add(rank)
        phases = ("hc", "dense-attention", "deltanet", "moe", "ple")
        missing = [phase for phase in phases
                   if not any(f"/{phase}" in label for label in labels)]
        if missing:
            raise SystemExit(f"Worker {rank} missing stage markers: {missing}")
        traces[str(path)] = {
            "rank": rank,
            "scheduled_rows": [int(step[2]) for step in steps],
            "kernel_intervals": len(device),
            "devices": sorted({int(event["args"]["device"]) for event in device
                               if "device" in event.get("args", {})}),
        }
    if captured_ranks != {0, 1, 2, 3}:
        raise SystemExit(f"Expected four worker ranks, got {sorted(captured_ranks)}")
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
