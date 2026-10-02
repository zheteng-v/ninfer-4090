#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Capacity>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.n != 5120 || w.k != x.ne[0] || w.padded_shape[1] != w.k ||
        residual_out.ne[0] != w.n || residual_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q5 linear_add K-split MMA: unsupported exact shape");
    }
    if (w.k == 6144) {
        launch_q5_ksplit_mma<5120, 6144, Capacity, Capacity, true>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_q5_ksplit_mma<5120, 17408, Capacity, Capacity, true>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: unsupported exact K");
    }
}

} // namespace

void q5_linear_add_ksplit_mma_residual_launch(const Tensor& x, const Weight& w,
                                              Tensor& residual_out, cudaStream_t stream) {
    if (x.ne[1] >= 2 && x.ne[1] <= 4) {
        launch<4>(x, w, residual_out, stream);
    } else if (x.ne[1] >= 5 && x.ne[1] <= 8) {
        launch<8>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: T must be in [2,8]");
    }
    CUDA_CHECK(cudaGetLastError());
}


} // namespace ninfer::ops::detail
