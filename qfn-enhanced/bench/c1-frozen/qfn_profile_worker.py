"""Reference-only stage markers for the installed XPU worker."""
from functools import wraps

from torch.profiler import record_function
from vllm.v1.worker.xpu_worker import XPUWorker


def mark(owner, method, label, scope):
    original = getattr(owner, method)

    @wraps(original)
    def wrapped(*args, **kwargs):
        with scope(label):
            return original(*args, **kwargs)

    setattr(owner, method, wrapped)


class QFNProfileWorker(XPUWorker):
    stage_scope = staticmethod(record_function)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        layers = 0
        for module in self.model_runner.model.modules():
            if type(module).__name__ != "Qwen4ExpDecoderLayer":
                continue
            label = f"qfn/r{self.rank}/l{module.layer_idx}"
            for name in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc = getattr(module, name)
                for method in ("mix", "combine", "combine_and_mix"):
                    mark(hc, method, f"{label}/hc/{name}/{method}", self.stage_scope)
            if module.layer_type == "linear_attention":
                mark(module.linear_attn, "forward", f"{label}/deltanet", self.stage_scope)
            else:
                mark(module.self_attn, "forward", f"{label}/dense-attention", self.stage_scope)
            mark(module.mlp, "forward", f"{label}/moe", self.stage_scope)
            if module.ple is not None:
                mark(module.ple, "forward", f"{label}/ple", self.stage_scope)
            layers += 1
        if layers != 48:
            raise RuntimeError(f"Rank {self.rank}: expected 48 QFN layers, saw {layers}")
        state = self.model_runner.model_state
        mark(state, "b70_pre_forward", f"qfn/r{self.rank}/ple-prepare", self.stage_scope)
        return result

    def execute_model(self, scheduler_output):
        rows = scheduler_output.total_num_scheduled_tokens
        with self.stage_scope(f"qfn/r{self.rank}/worker-step/rows={rows}"):
            return super().execute_model(scheduler_output)
