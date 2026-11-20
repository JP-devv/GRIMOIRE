"""Small XPU event capture on the critical prefill and first decode worker steps."""
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
from time import perf_counter_ns

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from qfn_profile_worker import QFNProfileWorker


class QFNBoundaryWorker(QFNProfileWorker):
    def load_model(self, *args, **kwargs):
        self.capture_rows = None
        self.captured = set()
        self.events = []
        result = super().load_model(*args, **kwargs)
        for name in ("all_reduce", "all_gather", "reduce_scatter", "all_gatherv", "all_to_all"):
            if hasattr(GroupCoordinator, name):
                self.mark_collective(name)
        return result

    def mark_collective(self, name):
        original = getattr(GroupCoordinator, name)

        @wraps(original)
        def wrapped(group, *args, **kwargs):
            if self.capture_rows is None:
                return original(group, *args, **kwargs)
            tensors = [value for value in (*args, *kwargs.values()) if torch.is_tensor(value)]
            logical_bytes = sum(value.numel() * value.element_size() for value in tensors)
            with self.stage_scope(f"qfn/r{self.rank}/collective/{name}/logical_bytes={logical_bytes}"):
                return original(group, *args, **kwargs)

        setattr(GroupCoordinator, name, wrapped)

    @contextmanager
    def stage_scope(self, label):
        if self.capture_rows is None:
            yield
            return
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        host_start = perf_counter_ns()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((label, start, end, host_start, perf_counter_ns()))

    def execute_model(self, scheduler_output):
        rows = scheduler_output.total_num_scheduled_tokens
        armed = rows == 2048 or (rows == 1 and 2048 in self.captured)
        if not armed or rows in self.captured:
            return super().execute_model(scheduler_output)
        self.capture_rows = rows
        self.events.clear()
        origin = torch.xpu.Event(enable_timing=True)
        origin.record()
        try:
            result = super().execute_model(scheduler_output)
            torch.xpu.synchronize()
            record = {
                "rank": self.rank,
                "scheduled_rows": rows,
                "stages": [{
                    "name": label,
                    "device_start_ms": origin.elapsed_time(start),
                    "device_end_ms": origin.elapsed_time(end),
                    "host_start_ns": host_start,
                    "host_end_ns": host_end,
                } for label, start, end, host_start, host_end in self.events],
            }
            path = Path(f"/work/traces/rank-{self.rank}-rows-{rows}.json")
            path.write_text(json.dumps(record) + "\n")
            self.captured.add(rows)
            return result
        finally:
            self.capture_rows = None
            self.events.clear()
