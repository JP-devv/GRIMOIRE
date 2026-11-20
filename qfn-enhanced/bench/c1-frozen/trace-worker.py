"""Read worker provenance and device activity from vLLM XPU profiler traces."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    traces = {}
    for path in sorted(args.trace_dir.rglob("*.pt.trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        device = [event for event in events
                  if event.get("cat") == "kernel" and event.get("dur", 0) > 0]
        if not device:
            raise SystemExit(f"Missing XPU kernel intervals in {path}")
        traces[str(path)] = {
            "kernel_intervals": len(device),
            "devices": sorted({int(event["args"]["device"]) for event in device
                               if "device" in event.get("args", {})}),
        }
    if len(traces) != 4:
        raise SystemExit(f"Expected four worker traces, got {len(traces)}")
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
