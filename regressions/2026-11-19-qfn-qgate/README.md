# Flash-Next q_proj Q/gate packing

`make test-qfn` feeds distinct sentinels per token, head, half, and within-head channel through the same row-span and offset functions as `mega_attn` and the production PP4/non-TP Q/gate split. It checks each raw TP4 slice against the installed vLLM loader. It then checks each split slice against PP4 and reconstructs the complete Q/gate result from all four ranks. Global `[all Q | all gate]` packing and swapped per-head halves must both fail on each rank.

The checkpoint's 12 full-attention `q_proj.weight` tensors are BF16 `[12288,2560]`. The inherited vLLM model mapper routes `q_proj` to `qkv_proj` with shard ID `q`. `Qwen3NextAttention` doubles the heads passed to `QKVParallelLinear` when output gating is active. `weight_loader_v2` hands the Q shard to `_ColumnvLLMParameter.load_qkv_weight`, which narrows at `rank * shard_size` and copies in order, without touching K or V. `_project_qkv_gate` then views those rows as heads and chunks each head into Q and gate before normalization. The fused projection+norm branch is CUDA-only in this installed source.

`capture_vllm.py` executes those installed loader and eager split method bodies on CPU. It reads the checkpoint headers, substituting sentinel columns for the actual weight data. For each of four VLLM ranks it saves the raw Q shard and runs the split on the **loaded** rows, then requires the concatenated Q/gate outputs to match an unsharded vLLM split. The binary fixtures are little-endian signed 32-bit values:

- `vllm-qgate.i32`: Q/gate pairs in token/head/channel order, SHA-256 `d8c3b6277a9bf440810ede472f022b5543b3fda64e5320ad07391e117ce6ab17`.
- `vllm-loader-rank-rows.i32`: raw Q projection rows in rank/token/row order, SHA-256 `a40b9ea02d5ec7f211f9c5128ee5eede6d3c40ea1a3c398ebc9206261af6a76b`.

Capture image: `b70-flash-next:tp2`, vLLM 0.30.0 build `gced6857af`; source SHA-256:

- `qwen3_next.py`: `4fa5112ac2bf7886d41d9e1f6c3da8793620d1ec49c961b31e9609459a6a7f7a`
- `linear.py`: `6c8ea995afb343d50d7a395eb50da14b5e2e5c33c246f8b93d8cc69673e2422c`
- `parameter.py`: `ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510`

External reference regeneration only: in the pinned vLLM image, run `python3 capture_vllm.py /path/to/site-packages/vllm /path/to/qwen38-w4a16-snapshot vllm-qgate-new.i32 --rank-rows vllm-loader-new.i32`. GRIMOIRE's C++ regression consumes the frozen binary fixtures. Building and testing GRIMOIRE do not run this script or require Python, PyTorch, or vLLM.
