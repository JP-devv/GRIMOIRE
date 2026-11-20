#pragma once

#include "b70/formats.hpp"

namespace b70 {

// Draft tensors are materialized in BF16 even when the engine's scratch is
// float. Keep these boundaries explicit; target HC uses its own FP32 path.
inline float qfn_mtp_bf16(float x) { return bf16_to_f32(f32_to_bf16(x)); }

inline float qfn_mtp_combine(float stream, float block, float gain = 1.0f) {
    return qfn_mtp_bf16(stream + qfn_mtp_bf16(block) * gain);
}

template <typename Exp>
inline float qfn_mtp_injection_gain(float logit, int streams, Exp exp) {
    return 2.0f / (1.0f + exp(-logit / float(streams)));
}

template <typename Exp>
inline float qfn_mtp_input_mix_silu(float down, int streams, Exp exp) {
    const float z = qfn_mtp_bf16(down) / float(streams);
    return qfn_mtp_bf16(z / (1.0f + exp(-z)));
}

} // namespace b70
