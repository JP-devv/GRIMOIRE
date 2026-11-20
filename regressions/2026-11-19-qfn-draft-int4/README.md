# QFN MTP draft INT4 storage

Run `make test-qfn`, or `make bin/test_qfn_draft_int4 && ./bin/test_qfn_draft_int4`.
The test needs neither the checkpoint nor a GPU. It checks all 16 distinct codes,
rejects interpreting the raw bytes as signed s4, and checks the first two 32-column
groups of expert 0's gate row 0 with their original, different BF16 scales.

The `qwen38-w4a16-mtp-snapshot` config declares symmetric, group-32,
`pack-quantized` INT4. The indexed draft shard
`mtp-routed-experts-int4.safetensors` has I32 `[640,320]` gate/up and `[2560,80]`
down for expert 0, with BF16 scales `[640,80]` and `[2560,20]` respectively.
Thus each output row is already contiguous in input-column order, with eight
codes per little-endian word and one BF16 scale per 32 input columns.

Reference checked on CPU against the installed `compressed-tensors` 0.17.0
`compressors/pack_quantized/helpers.py::unpack_from_int32` in `b70-flash-next:tp2`.
It reads the low nibble first and subtracts 8. The reference returned `-8…7`
for words `0x76543210, 0xfedcba98` and the 64 integers pinned in the test for
the real gate row. This helper is an on-disk format oracle, not the XPU MoE
loader. Installed vLLM 0.30.0 `Qwen4ExpMTP.load_weights` uses `AutoWeightsLoader`;
`compressed_tensors_moe_wna16.py::process_weights_after_loading` passes the
packed parameters to `fused_moe/oracle/int_wna16.py::_process_weights_xpu` via
the WNA16 converter. The XPU repack transposes the loader's intermediate
`[expert, packed_K, N]` tensors and views them as bytes (no nibble reorder).
The converter passes no zero points for XPU and documents that the kernel
handles the offset-8 encoding.

For Grimoire, copy the payload bytes per output row and the BF16 scales verbatim
and populate INT4 zero metadata with `8` per group. Do not pass the `0xff`
sentinel to `decode_int4` for those bytes. If a kernel requires two's-complement
s4 (`0xff`), an exact representation change is `byte ^ 0x88`; that flips the
high bit of each nibble, without requantization. This fixture does not qualify
any GPU kernel or the draft loader itself.
