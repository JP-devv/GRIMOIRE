"""Validate XPU event timing on a known operation before a model-worker run."""
import torch

value = torch.arange(4096, device="xpu", dtype=torch.float32)
torch.xpu.synchronize()
start, finish = [torch.xpu.Event(enable_timing=True) for _ in range(2)]
start.record()
output = torch.add(value, 3)
finish.record()
torch.xpu.synchronize()
elapsed = start.elapsed_time(finish)
assert elapsed > 0 and output[-1].item() == 4098
print(f"Known XPU add completed in {elapsed} ms")
