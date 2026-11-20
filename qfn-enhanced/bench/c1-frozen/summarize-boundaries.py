"""Summarize per-rank current-stream intervals without adding overlaps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


PHASE = re.compile(r"qfn/r\d+/(?:l\d+/)?(hc|dense-attention|deltanet|moe|ple|ple-prepare)(?:/|$)")
COLLECTIVE = re.compile(r"qfn/r\d+/collective/([a-z_]+)/logical_bytes=(\d+)$")


def union(intervals):
    end = 0.0
    span = 0.0
    for start, finish in sorted(intervals):
        if finish < start:
            raise ValueError("Negative device event interval")
        span += max(0.0, finish - max(start, end))
        end = max(end, finish)
    return span


def read_step(root, rank, rows):
    receipt = json.loads((root / f"rank-{rank}-rows-{rows}.json").read_text())
    if receipt["rank"] != rank or receipt["scheduled_rows"] != rows:
        raise ValueError("Rank and scheduled rows disagree")
    spans = defaultdict(list)
    payloads = defaultdict(int)
    host_step_start = None
    for stage in receipt["stages"]:
        name = stage["name"]
        interval = (stage["device_start_ms"], stage["device_end_ms"])
        phase = PHASE.match(name)
        collective = COLLECTIVE.fullmatch(name)
        if phase:
            spans[phase[1]].append(interval)
        if collective:
            spans["collective_stream_span"].append(interval)
            payloads[collective[1]] += int(collective[2])
        if name == f"qfn/r{rank}/worker-step/rows={rows}":
            spans["worker_step"].append(interval)
            host_step_start = stage["host_start_ns"]
    required = ("hc", "dense-attention", "deltanet", "moe", "ple", "worker_step")
    if host_step_start is None or any(not spans[phase] for phase in required):
        raise ValueError(f"Missing stage on rank {rank} at {rows} rows")
    return {"rank": rank, "rows": rows, "host_start_ns": host_step_start,
            "stream_union_ms": {phase: union(events) for phase, events in spans.items()},
            "observed_collective_api_logical_input_bytes": payloads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    ranks = [read_step(args.trace_dir, rank, rows) for rows in (2048, 1) for rank in range(4)]
    for row in ranks:
        earliest = min(other["host_start_ns"] for other in ranks if other["rows"] == row["rows"])
        row["rank_entry_skew_ms"] = (row["host_start_ns"] - earliest) / 1e6
    print(json.dumps(ranks, indent=2))


if __name__ == "__main__":
    main()
