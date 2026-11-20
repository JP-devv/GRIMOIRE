#include "kernels.hpp"
#include "qfn_small_moe.hpp"

namespace b70 {
namespace {
constexpr int kOutPerSubgroup = 2;
constexpr int kOutPerGroup = WG_SUBGROUPS * kOutPerSubgroup;
constexpr int kGroup = WG_SUBGROUPS * SG_SIZE;
static_assert(GEMV_EPL == 16 && GEMV_STEP == 256);

// Same packed bytes, BF16 scale and signed/affine convention as
// GemvStep<INT4,16>. Each lane loads them once for this expert output and
// applies the same pairwise FP32 arithmetic independently to each live row.
inline void dot(const QuantWeight& w, int out, int width, int lane, int count,
                const float* xs, float (&total)[kSmallMoeRows]) {
    const uint8_t* row = w.payload + int64_t(out) * w.row_bytes;
    for (int base = 0; base + GEMV_STEP <= width; base += GEMV_STEP) {
        const int k0 = base + lane * GEMV_EPL;
        const int64_t gi = int64_t(out) * w.row_scales + (k0 >> w.int4_gshift());
        const float scale = bf16_to_f32(static_cast<const bf16_t*>(w.scales)[gi]);
        const uint8_t zero = w.zeros[gi];
        float aq[kSmallMoeRows] = {}, ax[kSmallMoeRows] = {};
        #pragma unroll
        for (int i = 0; i < GEMV_EPL / 2; ++i) {
            const uint8_t byte = row[(k0 >> 1) + i];
            const int lo = byte & 15, hi = byte >> 4;
            const float a = float(zero == 0xff && (lo & 8) ? lo - 16 : lo);
            const float b = float(zero == 0xff && (hi & 8) ? hi - 16 : hi);
            #pragma unroll
            for (int r = 0; r < kSmallMoeRows; ++r) if (r < count) {
                const float x0 = xs[r * width + k0 + 2 * i];
                const float x1 = xs[r * width + k0 + 2 * i + 1];
                aq[r] = sycl::fma(a, x0, aq[r]);
                aq[r] = sycl::fma(b, x1, aq[r]);
                ax[r] += x0 + x1;
            }
        }
        #pragma unroll
        for (int r = 0; r < kSmallMoeRows; ++r) if (r < count)
            total[r] += (zero == 0xff ? aq[r] : aq[r] - float(zero) * ax[r]) * scale;
    }
    for (int k = width / GEMV_STEP * GEMV_STEP + lane; k < width; k += SG_SIZE) {
        const float value = w.at(out, k); // Down's 128-element tail uses the legacy scalar decode.
        #pragma unroll
        for (int r = 0; r < kSmallMoeRows; ++r) if (r < count)
            total[r] = sycl::fma(value, xs[r * width + k], total[r]);
    }
}

template<bool GateUp>
void project(sycl::queue& q, const QuantWeight& w, int rows,
             const SmallMoeRoutes* plan, const float* x, float* h, float* values) {
    constexpr int columns = GateUp ? kSmallMoeHidden : kSmallMoeInner;
    constexpr int outputs = GateUp ? kSmallMoeInner : kSmallMoeHidden;
    static_assert(outputs % kOutPerGroup == 0);
    constexpr int groups_per_batch = outputs / kOutPerGroup;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local(size_t(rows) * columns, cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(rows) * kSmallMoeTopK * groups_per_batch * kGroup, kGroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int wg = int(it.get_group_linear_id());
                const int batch_id = wg / groups_per_batch;
                if (batch_id >= plan->batches) return;  // Uniform guard before any SLM or weight access.
                const auto batch = plan->batch[batch_id];
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int lid = int(it.get_local_linear_id());
                const int out = (wg % groups_per_batch) * kOutPerGroup +
                                int(sg.get_group_id()[0]) * kOutPerSubgroup;
                for (int r = 0; r < batch.count; ++r) {
                    const float* src;
                    if constexpr (GateUp)
                        src = x + size_t(plan->slot[batch.begin + r] / kSmallMoeTopK) * kSmallMoeHidden;
                    else src = h + size_t(batch.begin + r) * kSmallMoeInner;
                    for (int j = lid; j < columns; j += kGroup) local[r * columns + j] = src[j];
                }
                sycl::group_barrier(it.get_group());
                const float* xs = local.template get_multi_ptr<sycl::access::decorated::no>().get();
                const int expert_base = batch.expert * (GateUp ? 2 * kSmallMoeInner : kSmallMoeHidden);
                #pragma unroll
                for (int i = 0; i < kOutPerSubgroup; ++i) {
                    float acc[kSmallMoeRows] = {};
                    dot(w, expert_base + out + i, columns, lane, batch.count, xs, acc);
                    if constexpr (GateUp) {
                        float up[kSmallMoeRows] = {};
                        dot(w, expert_base + kSmallMoeInner + out + i, columns, lane, batch.count, xs, up);
                        for (int r = 0; r < batch.count; ++r) {
                            const float g = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                            const float u = sycl::reduce_over_group(sg, up[r], sycl::plus<float>());
                            if (lane == 0)
                                h[size_t(batch.begin + r) * outputs + out + i] = (g / (1.0f + sycl::exp(-g))) * u;
                        }
                    } else {
                        for (int r = 0; r < batch.count; ++r) {
                            const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                            if (lane == 0) values[size_t(batch.begin + r) * outputs + out + i] = v;
                        }
                    }
                }
            });
    });
}
} // namespace

void launch_qfn_small_moe(sycl::queue& q, const MoeLayer& moe, int rows, int expert_begin,
                          const int32_t* ids, const float* weights, const float* x,
                          SmallMoeRoutes* plan, float* h, float* values, float* y) {
    q.single_task([=] { plan_small_moe_routes(ids, weights, rows, expert_begin, *plan); });
    project<true>(q, moe.gate_up, rows, plan, x, h, values);
    project<false>(q, moe.down, rows, plan, x, h, values);
    // A rank with no routes still has to publish a zero contribution for each row.
    // Fold reduced expert outputs in router order, not grouped projection order.
    q.parallel_for(sycl::range<1>(size_t(rows) * kSmallMoeHidden), [=](sycl::id<1> it) {
        const int element = int(it[0]);
        const int row = element / kSmallMoeHidden, out = element % kSmallMoeHidden;
        y[element] = fold_small_moe_routes(*plan, weights, row,
            [&](int route) { return values[size_t(route) * kSmallMoeHidden + out]; },
            [](float w, float value, float sum) { return sycl::fma(w, value, sum); });
    });
}
} // namespace b70
