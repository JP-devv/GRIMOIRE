#ifndef B70_QFN_MTP_EXPERTS_HPP
#define B70_QFN_MTP_EXPERTS_HPP

#include "safetensors.hpp"
#include "weights.hpp"

#include <array>
#include <memory>

namespace b70 {

struct QfnMtpExpert {
    PackedWeight gate, up, down;
};

// External Qwen4-Exp MTP layer 0: indexed, symmetric compressed-tensors INT4,
// row-major input groups of 32. Dense draft and target weights are not exposed.
class QfnMtpExperts {
public:
    // Validates all indexed routed projections and their declared shapes. No
    // packed values are read until read_expert(), which owns just its result.
    // Failure leaves the previous open checkpoint/result unchanged.
    bool open(const std::string& dir, std::string& err);
    bool read_expert(int id, QfnMtpExpert& out, std::string& err) const;

    int expert_count() const { return int(experts_.size()); }
    int hidden_size() const { return hidden_; }
    int intermediate_size() const { return inter_; }

private:
    struct Tensor {
        const SafeTensors* shard = nullptr;
        const STTensor* info = nullptr;
    };
    struct Projection {
        Tensor packed, scales;
    };
    int hidden_ = 0, inter_ = 0;
    std::vector<std::unique_ptr<SafeTensors>> shards_;
    std::vector<std::array<Projection, 3>> experts_; // gate, up, down
};

} // namespace b70
#endif
