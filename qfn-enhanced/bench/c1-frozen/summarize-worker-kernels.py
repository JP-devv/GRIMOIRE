"""Attribute device intervals via Kineto external IDs and report per-stage unions."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re


MARK = re.compile(r"qfn/r(\d+)/(?:l\d+/)?(hc|dense-attention|deltanet|moe|ple|ple-prepare)(?:/|$)")
STEP = re.compile(r"qfn/r(\d+)/worker-step/rows=(\d+)")
COLLECTIVE = re.compile(r"all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|oneccl|xccl",
                        re.IGNORECASE)


def union_ms(intervals):
    span = 0.0
    end = float("-inf")
    for begin, finish in sorted(intervals):
        span += max(0.0, finish - max(begin, end))
        end = max(end, finish)
    return span / 1000


def bounds(event):
    return event["ts"], event["ts"] + event["dur"]


def enclosing(cpu_event, candidates):
    start, end = bounds(cpu_event)
    thread = cpu_event["pid"], cpu_event["tid"]
    return [event for event in candidates
            if (event["pid"], event["tid"]) == thread
            and event["ts"] <= start and end <= event["ts"] + event["dur"]]


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = [event for event in json.load(stream)["traceEvents"]
                  if event.get("ph") == "X" and event.get("dur", 0) > 0]
    marks = [event for event in events if MARK.match(event.get("name", ""))]
    steps = [event for event in events if STEP.fullmatch(event.get("name", ""))]
    cpu_by_id = {}
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is not None:
            cpu_by_id.setdefault(external_id, event)
    by_step = {event["ts"]: defaultdict(list) for event in steps}
    unmatched = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        launch = cpu_by_id.get(event.get("args", {}).get("External id"))
        associated_steps = enclosing(launch, steps) if launch else []
        if len(associated_steps) != 1:
            unmatched.append(bounds(event))
            continue
        buckets = by_step[associated_steps[0]["ts"]]
        buckets["all_device"].append(bounds(event))
        parents = enclosing(launch, marks)
        if parents:
            stage = MARK.match(min(parents, key=lambda parent: parent["dur"])["name"])[2]
            buckets[stage].append(bounds(event))
        else:
            buckets["outside_stage_markers"].append(bounds(event))
        if COLLECTIVE.search(event["name"]) or COLLECTIVE.search(launch["name"]):
            buckets["collective_named_kernel"].append(bounds(event))
    return {
        "file": str(path),
        "unmatched_device_union_ms": union_ms(unmatched),
        "steps": [
            {"rank": int(match[1]), "scheduled_rows": int(match[2]),
             "device_union_ms": {stage: union_ms(intervals)
                                 for stage, intervals in by_step[step["ts"]].items()}}
            for step in sorted(steps, key=lambda event: event["ts"])
            for match in [STEP.fullmatch(step["name"])]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.trace], indent=2))


if __name__ == "__main__":
    main()
