"""Record actual multi-row schedules without XPU timing or fences."""
from vllm.v1.worker.xpu_worker import XPUWorker


class QFNReceiptWorker(XPUWorker):
    def execute_model(self, scheduler_output):
        rows = scheduler_output.total_num_scheduled_tokens
        if rows > 1:
            print(f"QFN schedule rank={self.rank} rows={rows}", flush=True)
        return super().execute_model(scheduler_output)
