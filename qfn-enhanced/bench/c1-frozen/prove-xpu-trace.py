"""Run with the installed reference image during the exclusive GPU window."""
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

trace = Path("/work/xpu-probe.json")
values = torch.arange(4096, device="xpu", dtype=torch.float32)
torch.xpu.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as profiler:
    with record_function("qfn_probe_known_xpu_add"):
        result = torch.add(values, 3)
        torch.xpu.synchronize()
assert result[-1].item() == 4098
profiler.export_chrome_trace(str(trace))
events = json.loads(trace.read_text())["traceEvents"]
assert any(event.get("name") == "qfn_probe_known_xpu_add" for event in events)
assert any(event.get("cat") == "kernel" and event.get("dur", 0) > 0 for event in events), (
    "The trace contains no device kernel interval"
)
print("XPU profiler captured a known device operation")
