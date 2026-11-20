"""Recover complete device, launch and stage events from a truncated Kineto export."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


CATEGORIES = {"kernel", "cpu_op", "xpu_runtime", "gpu_memcpy", "user_annotation"}


def recover(source, target):
    counts = Counter()
    in_events = False
    event_lines = []
    truncated = False
    with gzip.open(source, "rt") as reader, gzip.open(target, "wt") as writer:
        writer.write('{"traceEvents":[\n')
        try:
            for line in reader:
                if not in_events:
                    if '"traceEvents": [' in line:
                        in_events = True
                    continue
                if not event_lines and line.lstrip().startswith("]"):
                    break
                if not event_lines and not line.startswith("  {"):
                    continue
                event_lines.append(line)
                if line.rstrip() not in ("  },", "  }"):
                    continue
                raw = "".join(event_lines).rstrip().removesuffix(",")
                event_lines.clear()
                event = json.loads(raw)
                category = event.get("cat")
                if event.get("ph") == "X" and category in CATEGORIES:
                    if counts.total():
                        writer.write(",\n")
                    writer.write(json.dumps(event, separators=(",", ":")))
                    counts[category] += 1
        except EOFError:
            truncated = True
        writer.write('\n],"truncatedSource":' + json.dumps(truncated) + '}\n')
    return {"source": str(source), "target": str(target),
            "truncated_source": truncated, "complete_events": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("source", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir()
    print(json.dumps([recover(path, args.output_dir / path.name) for path in args.source], indent=2))


if __name__ == "__main__":
    main()
