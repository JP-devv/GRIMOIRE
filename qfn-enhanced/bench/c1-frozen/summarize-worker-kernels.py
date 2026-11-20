"""Attribute device intervals via Kineto external IDs and report per-stage unions."""
import argparse
from bisect import bisect_right
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


def by_thread(events):
    result = defaultdict(list)
    for event in sorted(events, key=lambda event: event["ts"]):
        result[event["pid"], event["tid"]].append(event)
    return {key: ([event["ts"] for event in thread], thread)
            for key, thread in result.items()}


def enclosing(cpu_event, candidates, earliest=float("-inf")):
    start, end = bounds(cpu_event)
    starts, events = candidates.get((cpu_event["pid"], cpu_event["tid"]), ((), ()))
    index = bisect_right(starts, start) - 1
    while index >= 0 and starts[index] >= earliest:
        event = events[index]
        if end <= event["ts"] + event["dur"]:
            return event
        index -= 1
    return None


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = [event for event in json.load(stream)["traceEvents"]
                  if event.get("ph") == "X" and event.get("dur", 0) > 0]
    marks = [event for event in events if MARK.match(event.get("name", ""))]
    steps = [event for event in events if STEP.fullmatch(event.get("name", ""))]
    step_index, stage_index = by_thread(steps), by_thread(marks)
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
        associated_step = enclosing(launch, step_index) if launch else None
        if associated_step is None:
            unmatched.append(bounds(event))
            continue
        buckets = by_step[associated_step["ts"]]
        buckets["all_device"].append(bounds(event))
        parent = enclosing(launch, stage_index, associated_step["ts"])
        if parent is not None:
            stage = MARK.match(parent["name"])[2]
            buckets[stage].append(bounds(event))
        else:
            buckets["outside_stage_markers"].append(bounds(event))
        if COLLECTIVE.search(event["name"]) or COLLECTIVE.search(launch["name"]):
            buckets["collective_named_kernel"].append(bounds(event))
    return {
        "file": str(path),
        "unmatched_device_union_ms": union_ms(unmatched),
        "steps": [
            {"rank": int(match[1]), "scheduled_rows": int(match[2]), "worker_host_span_ms": step["dur"] / 1000,
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
